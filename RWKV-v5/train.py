########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
    import pytorch_lightning as pl

    rank_zero_info("########## work in progress ##########")

    parser = ArgumentParser()

    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--wandb", default="", type=str)  # wandb project name. if "" then don't use wandb
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--random_seed", default="-1", type=int)
    parser.add_argument("--train_type", default="", type=str) # ""/"states"

    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="utf-8", type=str)   # xzl: binidx by default. "uint16" why special
    parser.add_argument("--vocab_size", default=0, type=int)  # vocab_size = 0 means auto (for char-level LM and .txt data)

    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument("--epoch_steps", default=1000, type=int)  # a mini "epoch" has [epoch_steps] steps
    parser.add_argument("--epoch_count", default=500, type=int)  # train for this many "epochs". will continue afterwards with lr = lr_final
    parser.add_argument("--epoch_begin", default=0, type=int)  # if you load a model trained for x "epochs", set epoch_begin = x
    parser.add_argument("--epoch_save", default=5, type=int)  # save the model every [epoch_save] "epochs"

    parser.add_argument("--micro_bsz", default=12, type=int)  # micro batch size (batch size per GPU)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)
    parser.add_argument("--pre_ffn", default=0, type=int)  # replace first att layer by ffn (sometimes better)      xzl: dark trick...
    parser.add_argument("--head_qk", default=0, type=int)  # my headQK trick        xzl:cf REAMDE
    parser.add_argument("--tiny_att_dim", default=0, type=int)  # tiny attention dim
    # xzl: can "shrink" att dim at specified layers... (cf model.py) not used in train script
    parser.add_argument("--tiny_att_layer", default=-999, type=int)  # tiny attention @ which layer         

    parser.add_argument("--lr_init", default=6e-4, type=float)  # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=-1, type=int)  # try 50 if you load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)  # use 0.999 when your model is close to convergence
    parser.add_argument("--adam_eps", default=1e-8, type=float)
    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--dropout", default=0, type=float) # try 0.01 / 0.02 / 0.05 / 0.1
    parser.add_argument("--weight_decay", default=0, type=float) # try 0.1 / 0.01 / 0.001
    parser.add_argument("--weight_decay_final", default=-1, type=float)

    # xzl: means what
    # pile mode -- stages of training? 1-init? >2 find saved model?
    parser.add_argument("--my_pile_version", default=1, type=int)  # my special pile version
    parser.add_argument("--my_pile_stage", default=0, type=int)  # my special pile mode
    parser.add_argument("--my_pile_shift", default=-1, type=int)  # my special pile mode - text shift
    parser.add_argument("--my_pile_edecay", default=0, type=int)
    parser.add_argument("--layerwise_lr", default=1, type=int)  # layerwise lr for faster convergence (but slower it/s)
    parser.add_argument("--ds_bucket_mb", default=200, type=int)  # deepspeed bucket size in MB. 200 seems enough
    # parser.add_argument("--cuda_cleanup", default=0, type=int)  # extra cuda cleanup (sometimes helpful)

    parser.add_argument("--my_sample_len", default=0, type=int)
    parser.add_argument("--my_ffn_shift", default=1, type=int)
    parser.add_argument("--my_att_shift", default=1, type=int)
    parser.add_argument("--head_size_a", default=64, type=int) # can try larger values for larger models
    parser.add_argument("--head_size_divisor", default=8, type=int)
    parser.add_argument("--my_pos_emb", default=0, type=int)
    parser.add_argument("--load_partial", default=0, type=int)
    parser.add_argument("--magic_prime", default=0, type=int)
    parser.add_argument("--my_qa_mask", default=0, type=int)    # xzl: qa task only??
    parser.add_argument("--my_random_steps", default=0, type=int)
    parser.add_argument("--my_testing", default='x052', type=str) # xzl:???  also  "x060" "g"?
    parser.add_argument("--my_exit", default=99999999, type=int)
    parser.add_argument("--my_exit_tokens", default=0, type=int)

    # xzl add 
    parser.add_argument("--svdfac", default=1, type=int) 
    parser.add_argument("--finetune", default=0, type=int)  # only finetune specific paras, freezing others
    parser.add_argument("--NoReLu", default=0, type=int) # use relu between decomposed weights?
    parser.add_argument("--NoDiag", default=1, type=int) # add diag to the weights? 
    parser.add_argument("--head_K", default=0, type=int)  # xzl: compress cls head as K clusters
    parser.add_argument("--load_token_cls", default="", type=str)  # token clusters, *.npy
    parser.add_argument("--lm_eval_0", default=1, type=int)  # run lm_eval before training/tuning starts, ensures lm_eval works 
    parser.add_argument("--lm_eval_n", default=1, type=int)  # run lm_eval on every chkpt we save
    parser.add_argument("--vram_mb", default=-1, type=int)  # detected gpu vram size, -1==uknown

    if pl.__version__[0]=='2':
        parser.add_argument("--accelerator", default="gpu", type=str)
        parser.add_argument("--strategy", default="auto", type=str)
        parser.add_argument("--devices", default=1, type=int)
        parser.add_argument("--num_nodes", default=1, type=int)
        parser.add_argument("--precision", default="fp16", type=str)
        parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    else:
        parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    ########################################################################################################

    import os, warnings, math, datetime, sys, time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    if "deepspeed" in args.strategy:
        import deepspeed
    from pytorch_lightning import seed_everything

    if args.random_seed >= 0:
        print(f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        seed_everything(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
    # os.environ["WDS_SHOW_SEED"] = "1"

    # xzl: override bsz based on gpu vram, model arch, and finetune or not 
    # e.g. [2,6,8,10] are micro_bsz for VRAM of ~12GB, ~24GB, ~40GB, ~80GB
    bsztable = {
        "L32-D2560-ctx2048-pretrain": [1,1,2,3],      # 3B OOM: bsz=4 for A100 80GB...
        "L24-D2048-ctx2048-pretrain": [1,1,4,8],      # 1B5 OOM: bsz=6 for A100 40GB ... 
        "L24-D1024-ctx2048-pretrain": [2,6,8,16],      # 04B. OOM: bsz=20 for A100 80GB; bsz=10 for A100 40GB ... why?
        "L24-D1024-ctx2048-finetune": [2,6,14,16], 
        "L12-D768-ctx2048-pretrain" : [2,8,18,24], 
        "L12-D768-ctx2048-finetune" : [2,10,20,30],
    }
    vram_idx = -1
    sss = "finetune" if args.finetune else "pretrain"
    bszkey = f"L{args.n_layer}-D{args.n_embd}-ctx{args.ctx_len}-{sss}"

    if (args.vram_mb > 10000 and args.vram_mb < 15000): 
        vram_idx = 0
    elif (args.vram_mb > 20000 and args.vram_mb < 30000): 
        vram_idx = 1
    elif (args.vram_mb > 40000 and args.vram_mb < 50000): 
        vram_idx = 2
    elif (args.vram_mb > 50000): 
        vram_idx = 3
    if vram_idx>=0 and bszkey in bsztable:
        args.micro_bsz = bsztable[bszkey][vram_idx] 

    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = 1.0
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
    args.max_epochs = -1  # continue forever
    args.betas = (args.beta1, args.beta2)
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    os.environ["RWKV_MY_TESTING"] = args.my_testing
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    os.environ["RWKV_TRAIN_TYPE"] = args.train_type
    if args.dim_att <= 0:
        args.dim_att = args.n_embd      # xzl: default attn dim ... same as embedding 
    if args.dim_ffn <= 0:
        if '-f4' in os.environ["RWKV_MY_TESTING"]:
            args.dim_ffn = int((args.n_embd * 4) // 32 * 32)
        else:
            args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size

    if args.data_type == "wds_img":
        args.run_name = f"v{args.my_img_version}-{args.my_img_size}-{args.my_img_bit}bit-{args.my_img_clip}x{args.my_img_clip_scale}"
        args.proj_dir = f"{args.proj_dir}-{args.run_name}"
    else:
        # args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd} F{args.svdfac}"
        # xzl 
        if "RUN_NAME" in os.environ: 
            args.run_name = os.environ["RUN_NAME"] # set by slurm-rva.sh or run.sh
        else: 
            args.run_name = f"L{args.n_layer} D{args.n_embd} F{args.svdfac} {args.my_testing}"  
    if not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)

    if args.my_pile_stage > 0:
        magic_prime_bak = args.magic_prime

        if args.my_pile_shift < 0:
            args.my_pile_shift = 0

        if magic_prime_bak > 0:
            args.magic_prime = magic_prime_bak
        if args.my_qa_mask == 2:
            args.epoch_count = 2 * args.magic_prime // 40320
        else:
            args.epoch_count = args.magic_prime // 40320

        args.epoch_steps = 40320 // args.real_bsz
        assert args.epoch_steps * args.real_bsz == 40320
        # if args.my_pile_stage == 2:
        #     assert args.lr_final == args.lr_init
        if args.my_pile_stage >= 2:  # find latest saved model
            list_p = []
            for p in os.listdir(args.proj_dir):
                if p.startswith("rwkv") and p.endswith(".pth"):
                    p = ((p.split("-"))[1].split("."))[0]
                    if p != "final":
                        if p == "init":
                            p = -1
                        else:
                            p = int(p)
                        list_p += [p]
            list_p.sort()
            if len(list_p) > 0:   # xzl
                max_p = list_p[-1]
            else:
                max_p = -1 # xzl
            if len(list_p) > 1:
                args.my_pile_prev_p = list_p[-2]  # in case max_p is corrupted
            if max_p == -1: # xzl: -init is init model file?
                args.load_model = f"{args.proj_dir}/rwkv-init.pth"
            else:
                args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
                if args.warmup_steps < 0:
                    if args.my_pile_stage == 2:
                        args.warmup_steps = 10
                    else:
                        args.warmup_steps = 30
            args.epoch_begin = max_p + 1

    samples_per_epoch = args.epoch_steps * args.real_bsz
    tokens_per_epoch = samples_per_epoch * args.ctx_len
    try:
        deepspeed_version = deepspeed.__version__
    except:
        deepspeed_version = None
        pass
    rank_zero_info(
        f"""
############################################################################
#
# RWKV-5 {args.precision.upper()} on {args.num_nodes}x{args.devices} {args.accelerator.upper()}, bsz {args.num_nodes}x{args.devices}x(Per GPU){args.micro_bsz}={args.real_bsz}, {args.strategy} {'with grad_cp' if args.grad_cp > 0 else ''}
#
# Data = {args.data_file} ({args.data_type}), ProjDir = {args.proj_dir}
#
# Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1} (will continue afterwards), save every {args.epoch_save} epoch
#
# Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
#
# Model = {args.n_layer} n_layer, {args.n_embd} n_embd, {args.ctx_len} ctx_len
#
# Adam = lr {args.lr_init} to {args.lr_final}, warmup {args.warmup_steps} steps, beta {args.betas}, eps {args.adam_eps}
#
# Found torch {torch.__version__}, recommend latest torch
# Found deepspeed {deepspeed_version}, recommend latest deepspeed
# Found pytorch_lightning {pl.__version__}, recommend 1.9.5
#
############################################################################
"""
    )
    rank_zero_info(str(vars(args)) + "\n")

    assert args.data_type in ["utf-8", "utf-16le", "numpy", "binidx", "dummy", "uint16"]

    if args.lr_final == 0 or args.lr_init == 0:
        rank_zero_info("\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n")

    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    if args.precision == "fp32":
        for i in range(10):
            rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
    if args.precision == "fp16":
        rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

    os.environ["RWKV_JIT_ON"] = "1"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0"

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"

    ########################################################################################################

    from src.trainer import train_callback, generate_init_weight
    from src.dataset import MyDataset

    train_data = MyDataset(args)
    args.vocab_size = train_data.vocab_size

    from src.model import RWKV
    model = RWKV(args)      # construct the model...

    #xzl: here init the model...        args.load_model: textual path to the model chkpt
    if len(args.load_model) == 0 or args.my_pile_stage == 1:  # shall we build the initial weights?
        init_weight_name = f"{args.proj_dir}/rwkv-init.pth"
        generate_init_weight(model, init_weight_name)  # save initial weights
        args.load_model = init_weight_name

    rank_zero_info(f"########## Loading {args.load_model}... ##########")
    try:
        load_dict = torch.load(args.load_model, map_location="cpu")
        load_keys = list(load_dict.keys())
        for k in load_keys:
            if k.startswith('_forward_module.'):
                load_dict[k.replace('_forward_module.','')] = load_dict[k]
                del load_dict[k]
    except:
        rank_zero_info(f"Bad checkpoint {args.load_model}")
        if args.my_pile_stage >= 2:  # try again using another checkpoint
            max_p = args.my_pile_prev_p
            if max_p == -1:
                args.load_model = f"{args.proj_dir}/rwkv-init.pth"
            else:
                args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
            args.epoch_begin = max_p + 1
            rank_zero_info(f"Trying {args.load_model}")
            load_dict = torch.load(args.load_model, map_location="cpu")

    # xzl: now we have a good model file, run lm_eval. 
    #       -- ensures lm_eval works prior to training 
    if args.lm_eval_0 == 1:
        # if 'x058' == os.environ["RWKV_MY_TESTING"]:
        #     from src.svd import recover_save
        #     eval_model_path = args.load_model.replace(".pth", "-recover.pth")
        #     recover_save(args.load_model.replace(".pth",""), eval_model_path.replace(".pth",""), 
        #                 args.n_layer, args.n_embd)
        # else:
        eval_model_path = args.load_model
        from src.run_lm_eval import do_eval
        from src.run_lm_eval import clean_cache
        res = do_eval(eval_model_path)
        clean_cache() # otherwise next run_lm_eval will cache the results
        import json
        print(json.dumps(res)+'\n') # just write to console

    # xzl: allow the ckpt file to lack certain params, in which case just 
    #   keep the model's params as is (what values???
    if args.load_partial == 1:
        load_keys = load_dict.keys()
        for k in model.state_dict():
            if k not in load_keys:
                if "head_l2" in k:
                    continue  # will process in "build head_l2" loop below
                load_dict[k] = model.state_dict()[k]
                # if "head_l1" in k or "head_l2" in k:        # xzl: cls head
                if "head_l1.weight" in k:
                    if args.vocab_size > args.n_embd:
                        scale = 0.5 * math.sqrt(args.vocab_size / args.n_embd)
                    else:
                        scale = 0.5
                    torch.nn.init.orthogonal_(load_dict[k], gain=scale)
                if "head_l1fc" in k:
                    torch.nn.init.kaiming_uniform_(load_dict[k], nonlinearity='relu')
                if "diag" in k: 
                    scale = -1e-4
                    torch.nn.init.uniform_(load_dict[k], a=scale, b=-scale)
        
        # build head_l2, but by splitting the original cls head weights
        orghead = load_dict['head.weight']   # must exist
        for cls in range(0, args.head_K):
            k = f'head_l2.{cls}.weight'
            if k in load_keys:
                continue
            idx = torch.tensor(model.clusters[cls], device=orghead.device)
            ww = orghead[idx]
            load_dict[k] = ww  #save it 
    
    # below: load from state_dict (tensors) to model 
    # model.load_state_dict(load_dict, strict=False)
    model.load_state_dict(load_dict, strict=True)

    if pl.__version__[0]=='2':
        trainer = Trainer(accelerator=args.accelerator,strategy=args.strategy,devices=args.devices,num_nodes=args.num_nodes,precision=args.precision,
        logger=args.logger,callbacks=[train_callback(args)],max_epochs=args.max_epochs,check_val_every_n_epoch=args.check_val_every_n_epoch,num_sanity_val_steps=args.num_sanity_val_steps,
        log_every_n_steps=args.log_every_n_steps,enable_checkpointing=args.enable_checkpointing,accumulate_grad_batches=args.accumulate_grad_batches,gradient_clip_val=args.gradient_clip_val)
    else:
        trainer = Trainer.from_argparse_args(
            args,
            callbacks=[train_callback(args)],
        )
        # ^^ trainer.strategy also constructed

    # xzl after loading model, print out info:  layers, params shapes, etc. 
    if trainer.global_rank == 0:
        print("Dump model arch ....")
        for n in model.state_dict():            # xzl: n: para name, can be used as key        cf https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html             
            # if "blocks." in n and not "blocks.0." in n: 
            if not "blocks.0." in n: # only print params in "blocks"
                continue # xzl: only print layer0, less cluter ....            
            shape = model.state_dict()[n].shape
            shape = [i for i in shape if i != 1]
            if len(shape) > 2:
                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(shape[2]).ljust(5)} {n}")
            elif len(shape) > 1:
                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)}       {n}")
            elif len(shape) > 0:
                print(f"{str(shape[0]).ljust(5)}             {n}")
                # print(f"{str(shape).ljust(5)}             {n}")
        print("(Omit other layers...")
    if "deepspeed" in args.strategy:
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = args.ds_bucket_mb * 1000 * 1000

    # xzl: NB config strategy here.....
    # breakpoint()    

    # must set shuffle=False, persistent_workers=False (because worker is in another thread)
    data_loader = DataLoader(train_data, shuffle=False, pin_memory=True, 
                                batch_size=args.micro_bsz, 
                                num_workers=0,      # =0 for debugging, =1 normal (=1 failed on mps, why??
                                persistent_workers=False, drop_last=True)
    # xzl: above: DataLoader decides how data is fed into trainer, which goes to callbacks in model.py


    if args.finetune or args.head_K > 1:
        if args.finetune:
            tunepara = [".att.receptance1.", ".att.key1.", ".att.value1.", ".att.gate1.", 
                        ".att.receptance2.", ".att.key2.", ".att.value2.", ".att.gate2.", 
                        ".ffn.receptance1.", ".ffn.receptance2.", 
                        # "blocks.23.",
                        # ".att.output.",
                        # "head.weight",
                        # "ln_x", "ln1", "ln2", 
                        "ln_out",   # must include this otherwise loss has no grad_fn (below).... 
                        ]
        if args.head_K > 1:
            # tunepara = ["head_l1", "head_l2"]
            tunepara = ["head_l1"]
            #tunepara = ["head_l1fc"]

        model.requires_grad_(False)    #xzl this seems a must
        for pname, param in model.named_parameters():
            for tp in tunepara:
                if tp in pname:
                    param.requires_grad = True
                    if "blocks.0" in pname:
                        ppname = pname.replace("blocks.0.","")
                        print(f"will train: {ppname}")
                    else:
                        print(f"will train: {pname}")
                # else:
                #     param.requires_grad = False
    # if args.train_type == 'states':
    #     model.requires_grad_(False)
    #     for name, module in model.named_modules():
    #         for pname, param in module.named_parameters():
    #             if pname.endswith('.time_state') and pname.startswith('blocks.'):
    #                 print(pname)
    #                 param.requires_grad = True

    
    # sanity check weights...
    # param_tensor = model.head_l1fc1.weight
    # l2_norm = torch.norm(param_tensor, p=2)
    # print("head_l1fc1 L2 Norm:", l2_norm.item())
    # param_tensor = model.head_l1fc2.weight
    # l2_norm = torch.norm(param_tensor, p=2)
    # print("head_l1fc2 L2 Norm:", l2_norm.item())
    # breakpoint()

    trainer.fit(model, data_loader)
