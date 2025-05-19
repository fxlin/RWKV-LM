#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;


// _k global ptr, shared by all blocks 
// k_ (or __shared__) shared within a cudablk (a head)
// k (thr private, on registers

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const float *__restrict__ _w, const F *__restrict__ _u,
                               F *__restrict__ const _y) 
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _w += h*_N_;
    _u += h*_N_;        

    __shared__ float r[_N_], k[_N_], u[_N_], w[_N_];    // a head, block-local var 
    float state[_N_] = {0};  

    //  copy u,w from global tensor to this head (NB _w _u already shifted above
    __syncthreads();    
    w[i] = _w[i];
    u[i] = float(_u[i]);
    __syncthreads();

    // the whole loop belo: serial scan across timesteps in one sequence
    //  t: element index in a (B,T,C) tensor
    //      t+=C .. each iteration processes 1 tiemstemp? (eg t+=1024)
    //      this thr goes to the next token in the seq??
    for (int t = b*T*C + h*_N_ + i; t < (b+1)*T*C + h*_N_ + i; t += C)
    {
        __syncthreads();       
        r[i] = float(_r[t]);   
        k[i] = float(_k[t]);
        __syncthreads();       

        const float v = float(_v[t]);
        float y = 0;  

        // _N_: go through each element in *this* head (in r,k,w,u...)
        //   b/c this thr maintains state 1x_N_ (this head's state _N_x_N_
        // (+=4 b/c of float4)
        // essetinaly, decompose s@w, k@v ... into scalar mul...

        #pragma unroll
        for (int j = 0; j < _N_; j+=4)
        {
            const float4& r_ = (float4&)(r[j]);
            const float4& k_ = (float4&)(k[j]);
            const float4& w_ = (float4&)(w[j]);
            const float4& u_ = (float4&)(u[j]);
            float4& s = (float4&)(state[j]);
            float4 x;
            
            x.x = k_.x * v;
            x.y = k_.y * v;
            x.z = k_.z * v;
            x.w = k_.w * v;
            
            //      elementwise mac.. then accu along row ('
            y += r_.x * (u_.x * x.x + s.x);
            y += r_.y * (u_.y * x.y + s.y);
            y += r_.z * (u_.z * x.z + s.z);
            y += r_.w * (u_.w * x.w + s.w);

            s.x = s.x * w_.x + x.x;
            s.y = s.y * w_.y + x.y;
            s.z = s.z * w_.z + x.z;
            s.w = s.w * w_.w + x.w;
        }
        _y[t] = F(y);
    }
}

template <typename F>
__global__ void kernel_backward(const int B, const int T, const int C, const int H,
    const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const float *__restrict__ _w, const float *__restrict__ __w, const F *__restrict__ _u, const F *__restrict__ const _gy,
    F *__restrict__ const _gr, F *__restrict__ const _gk, F *__restrict__ const _gv, F *__restrict__ const _gw, F *__restrict__ const _gu)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _w += h*_N_;        
    _u += h*_N_;
    __w += h*_N_;

    __shared__ float w_[_N_], u_[_N_];
    __shared__ float r[_N_], k[_N_], v[_N_], gy[_N_];
    __syncthreads();
    w_[i] = _w[i];
    u_[i] = float(_u[i]);
    __syncthreads();

    const float w = w_[i];
    const float ww = __w[i];
    const float u = u_[i];

    float state[_N_] = {0}, saaaa[_N_] = {0}, sbbbb[_N_] = {0}, scccc[_N_] = {0}, sdddd[_N_] = {0};

    float gw = 0, gu = 0;
    const int t000 = b*T*C + h*_N_ + i;
    const int t111 = (b+1)*T*C + h*_N_ + i;  
    const int t222 = t111 - 2*C;

    for (int t = t000; t < t111; t += C)
    {
        __syncthreads();    
        v[i] = float(_v[t]);
        gy[i] = float(_gy[t]); 
        __syncthreads();

        const float k = float(_k[t]);
        float gr = 0, gu_ = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = state[j];
            float x = k * v[j]; 

            gr += (u * x + s) * gy[j];  
            gu_ += x * gy[j];
            s = s * w + x;
        }
        _gr[t] = F(gr);                 // output gr
        gu += float(_r[t]) * gu_;
    }
    _gu[b*C + h*_N_ + i] = F(gu);       // output gu....(shape B,C) grad on weights
    
    for (int t = t000; t < t222; t += C) 
    {
        __syncthreads();
        v[i] = float(_v[t]);
        gy[i] = float(_gy[t + 2*C]); 
        __syncthreads();

        const float k = float(_k[t]);
        float gw_ = 0;
        
        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = saaaa[j];
            float& s2 = sbbbb[j];
            float x = k * v[j];
            
            float tmp = w * (x + s);
            s = tmp;
            s2 = tmp + w * s2;
            gw_ += s2 * gy[j];      
        }
        gw += float(_r[t + 2*C]) * gw_;
    }    
    _gw[b*C + h*_N_ + i] = F(ww * gw);  // output gw, .. shape B,C weights

    // gk... 1 token less, grad flow from gy, back in time (from seq end
    //  state: scccc
    for (int t = t111 - C; t >= t000; t -= C)   
    {
        __syncthreads();
        v[i] = float(_v[t]);
        gy[i] = float(_gy[t]);
        __syncthreads();

        const float rr = float(_r[t]);
        float gk = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = scccc[j];
            float x = rr * gy[j];
            
            gk += (u * x + s) * v[j];
            s = x + s * w;
        }
        _gk[t] = F(gk);
    }

    // gv... 1 token less, back in time...
    //      state: sdddd
    for (int t = t111 - C; t >= t000; t -= C)   
    {
        __syncthreads();
        r[i] = float(_r[t]);
        k[i] = float(_k[t]);
        __syncthreads();

        const float gyy = float(_gy[t]);
        float gv = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = sdddd[j];
            float x = gyy * r[j];
            
            gv += (u_[j] * x + s) * k[j];
            s = x + s * w_[j];
        }
        _gv[t] = F(gv);
    }
}

void cuda_forward(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, float *w, bf16 *u, bf16 *y)
{
    assert(H*_N_ == C);
    assert(_N_%4 == 0);
    kernel_forward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, u, y);
}

void cuda_backward(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, float *w, float *ww, bf16 *u, bf16 *gy, bf16 *gr, bf16 *gk, bf16 *gv, bf16 *gw, bf16 *gu)
{
    assert(H*_N_ == C);
    assert(_N_%4 == 0);
    kernel_backward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, ww, u, gy, gr, gk, gv, gw, gu);
}
