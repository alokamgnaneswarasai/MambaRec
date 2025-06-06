import os
import time
import torch
import argparse
import wandb
import warnings
from model import *
from utils import *
from tqdm import tqdm
from slender_mamba.ops.Bitembedding import replace_embeddings_in_pytorch_model
from slender_mamba.ops.Bitembedding import replace_linears_in_pytorch_model

warnings.filterwarnings('ignore')

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--eval_neg_sample', default=1000, type=int)
parser.add_argument('--backbone', default='sas', type=str)
parser.add_argument('--name', default='mamba', type=str)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--maxlen', default=20, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=501, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda:4', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()



# print the selected arguments
print("Arguments:")
for k, v in sorted(vars(args).items(), key=lambda x: x[0]):
    print(k, v)
print()


# wandb.init(
#      # set the wandb project where this run will be logged
#      project="Kuairand",
#      entity="ssmrec",
#      name=args.name,

#      # track hyperparameters and run metadata
#      config={
#          "learning_rate": args.lr,
#          "architecture": args.backbone,
#          "dataset": args.dataset,
#          "epochs": args.num_epochs,
#      }
#  )
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # global dataset
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    print(len(user_train))
    num_batch = len(user_train) // args.batch_size  # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    if args.backbone == 'mamba':
        model = MambaRec(usernum, itemnum, args).to(args.device)  # no ReLU activation in original SASRec implementation?
       
    elif args.backbone == 'qmamba':
        model = QuantizedMambaRec(usernum, itemnum, args).to(args.device)  # no ReLU activation in original SASRec implementation?
        # model = MambaRec(usernum, itemnum, args).to(args.device)  # no ReLU activation in original SASRec implementation?
        print("Replacing linears")
        
        # check how many total linear layer parameters are in the model recursively
        print(model)
        for name, module in model.named_modules():
            print(f"Module: {name}, Type: {type(module)}")
        print("Total linear layer parameters: ", sum(sum(p.numel() for p in module.parameters()) for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)))
        replace_linears_in_pytorch_model(model)
        # print("Replacing embeddings")
        # replace_embeddings_in_pytorch_model(model)
        
    elif args.backbone == 'mamba2':
        model = Mamba2Rec(usernum, itemnum, args).to(args.device)
        
        
    elif args.backbone == 'sas':
        model = SASRec(usernum, itemnum, args).to(args.device)  # no ReLU activation in original SASRec implementation?
    elif args.backbone == 'linrec':
        model = LinRec(usernum, itemnum, args).to(args.device)
        
    elif args.backbone == 'sasmamba':
        model = SASmambaRec(usernum, itemnum, args).to(args.device)
        
    elif args.backbone == 'gatingsasmamba':
        model = GatingSASmambaRec(usernum, itemnum, args).to(args.device)
        
    elif args.backbone == 'jamba':
        model = Jamba4Rec(usernum, itemnum, args).to(args.device)
        
    elif args.backbone == 'hsas':
        model = HierarchicalSASRec(usernum, itemnum, args).to(args.device)
        
    elif args.backbone == 'moemamba':
        model = MoEMambaRec(usernum, itemnum, args).to(args.device)
        
    elif args.backbone == 'samba':
        model = SAMBA4Rec(usernum, itemnum, args).to(args.device)
        
    elif args.backbone =='hourglass':
        model = HourglassTransformer(usernum,itemnum,args).to(args.device)
        
    elif args.backbone =='transformerxl':
        model = TransformerXLEncoder(usernum,itemnum,args).to(args.device)
        
    elif args.backbone =='compressive':
        model = CompressiveTransformerRec(usernum,itemnum,args).to(args.device)
        
    elif args.backbone =='bimamba':
        model = BiMambaRec(usernum,itemnum,args).to(args.device)
        
    elif args.backbone == 'wsas':
        model = WindowSASRec(usernum, itemnum, args).to(args.device)
        print("Quantizing model")
        # replace_embeddings_in_pytorch_model(model)
        replace_linears_in_pytorch_model(model)
        
    elif args.backbone =='owsas':
        model = optimizedWindowSASRec(usernum,itemnum, args).to(args.device)
        
    elif args.backbone=='localsas':
        model = localSASRec(usernum, itemnum, args).to(args.device) 
        
    elif args.backbone == 'localmamba':
        model = localmamba(usernum,itemnum,args).to(args.device)
    
    elif args.backbone == 'qlocalmamba':
        model = localmamba(usernum,itemnum,args).to(args.device)
        print("Quantizing model")
        print("Replacing linears")
        replace_linears_in_pytorch_model(model)
        
    elif args.backbone == 'localmamba2':
        model = localmamba2(usernum,itemnum,args).to(args.device)
        
    elif args.backbone =='stackedmamba':
        model = StackedMambaLocalAttention(usernum,itemnum,args).to(args.device)
               
    elif args.backbone =="mamba4rec+local":
        model = mamba4rec(usernum,itemnum,args).to(args.device)
        
    elif args.backbone == 'noiselocalmamba':
        model = noiselocalmamba(usernum,itemnum,args).to(args.device)
        
    elif args.backbone == 'usernoiselocalmamba':
        model = usernoiselocalmamba(usernum,itemnum,args).to(args.device)
        
    elif args.backbone == 'dynamicusernoiselocalmamba':
        model = dynamicusernoiselocalmamba(usernum,itemnum,args).to(args.device)
        
    elif args.backbone == 'queryusernoiselocalmamba':
        model = queryusernoiselocalmamba(usernum,itemnum,args).to(args.device)
        
                   
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # just ignore those failed init layers
    
    
    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    print('parameters_count:', count_parameters(model))
    model.train()  # enable model training

    epoch_start_idx = 1


    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    inference_time = 0.0
    best_HR_10_val = 0
    best_epoch = 0
    # mem = None
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break  # just to decrease identition
        for step in (range(num_batch)) :  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            
            t0 = time.time()
            
            # pos_logits, neg_logits,mem = model(u, seq, pos, neg)
            # print("model is going to run")
            pos_logits, neg_logits = model(u, seq, pos, neg)
            t1 = time.time()
            T += (t1 - t0) * 1000
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                                   device=args.device)
           
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0) 
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats(args.device)
            # print("loss is going to run")
            t1 = time.time()
            loss.backward()
            t2 = time.time()
            # print(f"backward runs for {(t2 - t1)} seconds")
            
            if epoch % 99 == 1 and step == 1:
                print(f"Current Memory Allocated: {torch.cuda.memory_allocated(args.device) / 1024**2:.2f} MB")
                print(f"Current Memory Cached: {torch.cuda.memory_reserved(args.device) / 1024**2:.2f} MB")
                print(f"Peak Memory Cached: {torch.cuda.max_memory_reserved(args.device) / 1024**2:.2f} MB")
            # print(f"Peak Memory Allocated: {torch.cuda.max_memory_allocated(args.device) / 1024**2:.2f} MB")
            
            t1 = time.time()
            T += (t1 - t0) * 1000
            adam_optimizer.step()
        print("loss in epoch {} iteration {}: {}".format(epoch, step,
                                                         loss.item()))  # expected 0.4~0.6 after init few epochs
        
        
    
        if not os.path.exists(f'saved_models/{args.dataset}/{args.backbone}'):
            os.makedirs(f'saved_models/{args.dataset}/{args.backbone}')
            
        if epoch % 5 == 0:
            model.eval()
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            inference_time += t_test[6]
            t_valid = evaluate_valid(model, dataset, args)
            print(
                'epoch:%d, time: %f(s), valid (NDCG@5: %.4f, HR@5: %.4f, NDCG@10: %.4f, HR@10: %.4f, NDCG@20: %.4f, HR@20: %.4f), test (NDCG@5: %.4f, HR@5: %.4f, NDCG@10: %.4f, '
                'HR@10: %.4f, NDCG@20: %.4f, HR@20: %.4f)' % (
                    epoch, T, t_valid[0], t_valid[1], t_valid[2], t_valid[3], t_valid[4], t_valid[5],
                    t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5]))

            if epoch%100==0 and t_valid[3] > best_HR_10_val:
                best_HR_10_val = t_valid[3]
                best_epoch = epoch
                 
                folder = f'saved_models/{args.dataset}/{args.backbone}'
                # fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.eval_neg_sample={}.pth'
                fname='{}.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.eval_neg_sample={}.pth'
                # fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units,
                #                      args.maxlen, args.eval_neg_sample)
                fname=fname.format(args.dataset,epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen, args.eval_neg_sample)
                torch.save(model.state_dict(), os.path.join(folder, fname))

            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            model.train()

        if epoch == args.num_epochs:
            folder = f'saved_models/{args.dataset}/{args.backbone}'
            fname = '{}.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.eval_neg_sample={}_final.pth'
            fname = fname.format(args.dataset, args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units,
                                 args.maxlen, args.eval_neg_sample)
            torch.save(model.state_dict(), os.path.join(folder, fname))
            
    print("Training time :", T)
    f.close()
    sampler.close()
    print("Done")
    


