import time
import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
import argparse
import json
#import torchvision.transforms as transforms
from data.LEVIR_CC.LEVIRCC import LEVIRCCDataset_video
from model.model_encoder import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer_video
from utils import *
from model.video_encoder import Video_encoder,Sty_fusion


def main(args):
    """
    Training and validation.
    """
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if os.path.exists(args.savepath)==False:
        os.makedirs(args.savepath)
    best_bleu4 = 0.4  # BLEU-4 score right now
    start_epoch = 0
    
    # CSV Logging initialization
    csv_file = os.path.join(args.savepath, 'training_progress.csv')
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Step', 'Train_Loss', 'Val_Loss', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'CIDEr'])

    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)
    # Initialize / load checkpoint  2644246
    if args.checkpoint is None:      
        video_encoder=Video_encoder()
        sty_fusion=Sty_fusion()
        sty_fusion_optimizer = torch.optim.Adam(sty_fusion.parameters(), lr=args.encoder_lr, weight_decay=1e-2)
        parameters = []
        for name, param in video_encoder.named_parameters():
            if 'att_liner' in name:
                parameters.append({'params': param, 'lr': 1e-6})
        print("Trainable layers in Video_encoder:")
        for name, param in video_encoder.named_parameters():
            if param.requires_grad:
                print(name)
        video_encoder_optimizer = torch.optim.Adam(parameters, lr=args.encoder_lr, weight_decay=1e-2)

        decoder = DecoderTransformer_video(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim, vocab_size=len(word_vocab), max_lengths=args.max_length, word_vocab=word_vocab, n_head=args.n_heads,
                                    n_layers= args.decoder_n_layers, dropout=args.dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                            lr=args.decoder_lr)
    else:
        video_encoder=Video_encoder()
        decoder = DecoderTransformer_video(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim, vocab_size=len(word_vocab), max_lengths=args.max_length, word_vocab=word_vocab, n_head=args.n_heads,
                                    n_layers= args.decoder_n_layers, dropout=args.dropout)
        sty_fusion=Sty_fusion()
        sty_fusion_optimizer = torch.optim.Adam(sty_fusion.parameters(), lr=args.encoder_lr, weight_decay=1e-2)
        checkpoint = torch.load(args.checkpoint)
        video_encoder.load_state_dict(checkpoint['video_encoder_dict'])
        parameters = []
        for name, param in video_encoder.named_parameters():
            if 'att_liner' in name:
                parameters.append({'params': param, 'lr': 1e-6})
        print("Trainable layers in Video_encoder:")
        for name, param in video_encoder.named_parameters():
            if param.requires_grad:
                print(name)
        decoder.load_state_dict(checkpoint['decoder_dict'])

        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                            lr=args.decoder_lr)
        video_encoder_optimizer = torch.optim.Adam(parameters, lr=args.encoder_lr, weight_decay=1e-2)
    # Move to GPU, if available
    video_encoder = video_encoder.to(dtype=torch.bfloat16).cuda()
    decoder = decoder.to(dtype=torch.bfloat16).cuda()
    sty_fusion = sty_fusion.to(dtype=torch.bfloat16).cuda()
    # Loss function
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # Custom dataloaders
    if args.data_name == 'LEVIR_CC':
        train_loader = data.DataLoader(
            LEVIRCCDataset_video(args.data_folder, args.list_path, 'train', args.token_folder,
                                 args.vocab_file, args.max_length, args.allow_unk, if_mask=True, mask_mode=args.mode),
            batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = data.DataLoader(
            LEVIRCCDataset_video(args.data_folder, args.list_path, 'val', args.token_folder,
                                 args.vocab_file, args.max_length, args.allow_unk, if_mask=True, mask_mode=args.mode),
            batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    else:
        raise ValueError(f"Sadece LEVIR_CC desteklenmektedir. Sağlanan data_name: {args.data_name}")

    decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=5, gamma=0.5)
    l_resizeA = torch.nn.Upsample(size = (256, 256), mode ='bilinear', align_corners = True)
    l_resizeB = torch.nn.Upsample(size = (256, 256), mode ='bilinear', align_corners = True)
    index_i = 0
    step_checkpoint_paths = []  # Son 3 ara kaydı takip etmek için
    hist = np.zeros((args.num_epochs * len(train_loader), 3))
    # Epochs
    
    for epoch in range(start_epoch, args.num_epochs):        
        # Batches
        for id, (video_tensor, _, _, token, token_len, _,mask) in enumerate(train_loader):
            start_time = time.time()
            decoder.train()  
            video_encoder.train() 
            sty_fusion.train()
            decoder_optimizer.zero_grad()
            video_encoder_optimizer.zero_grad()
            sty_fusion_optimizer.zero_grad()

            video_tensor=video_tensor.to(dtype=torch.bfloat16, device='cuda')
            mask=mask.to(dtype=torch.bfloat16, device='cuda')
            vedie_emb,_=video_encoder(video_tensor)
            vedie_emb=sty_fusion(vedie_emb,mask)

            token = token.squeeze(1).cuda()
            token_len = token_len.cuda()

            scores, caps_sorted, decode_lengths, sort_ind = decoder(vedie_emb, token, token_len)
            
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            loss = criterion(scores, targets)
            # Back prop.
            loss.backward()
            # Clip gradients
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(decoder.parameters(), args.grad_clip)
                # torch.nn.utils.clip_grad_value_(encoder_trans.parameters(), args.grad_clip)

            # Update weights  
            decoder_optimizer.step()         
            video_encoder_optimizer.step()           
            sty_fusion_optimizer.step()


            # Keep track of metrics     
            hist[index_i,0] = time.time() - start_time #batch_time        
            hist[index_i,1] = loss.item() #train_loss
            hist[index_i,2] = accuracy(scores, targets, 5) #top5
            index_i += 1

            # Ara kayıt mekanizması (her save_steps adımda bir)
            if index_i % args.save_steps == 0:
                os.makedirs('./checkpoints', exist_ok=True)
                step_ckpt_name = f'checkpoint_step_{index_i}.pth'
                step_ckpt_path = os.path.join('./checkpoints', step_ckpt_name)
                torch.save({
                    'step': index_i,
                    'epoch': epoch,
                    'video_encoder_dict': video_encoder.state_dict(),
                    'sty_fusion_dict':    sty_fusion.state_dict(),
                    'decoder_dict':       decoder.state_dict(),
                }, step_ckpt_path)
                print(f'[ARA KAYIT] {step_ckpt_path} kaydedildi.')
                # Sadece son 3 ara kaydı tut, eskisini sil
                step_checkpoint_paths.append(step_ckpt_path)
                if len(step_checkpoint_paths) > 3:
                    old = step_checkpoint_paths.pop(0)
                    if os.path.exists(old):
                        os.remove(old)
                        print(f'[ARA KAYIT] Eski checkpoint silindi: {old}')
            # Print status
            if index_i % args.print_freq == 0:
                cur_train_loss = np.mean(hist[index_i-args.print_freq:index_i-1,1])
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time: {3:.3f}\t'
                    'Loss: {4:.4f}\t'
                    'Top-5 Accuracy: {5:.3f}'.format(epoch, index_i, args.num_epochs*len(train_loader),
                                            np.mean(hist[index_i-args.print_freq:index_i-1,0])*args.print_freq,
                                            cur_train_loss,
                                            np.mean(hist[index_i-args.print_freq:index_i-1,2])))

                # Hızlı (Otomatik) Validation Loss Hesabı
                video_encoder.eval()
                sty_fusion.eval()
                decoder.eval()
                v_loss_sum = 0.0
                v_steps = 0
                with torch.no_grad():
                    for _, (v_tensor, _, _, v_tok, v_tok_len, _, v_mask) in enumerate(val_loader):
                        v_tensor = v_tensor.to(dtype=torch.bfloat16, device='cuda')
                        v_mask = v_mask.to(dtype=torch.bfloat16, device='cuda')
                        v_emb, _ = video_encoder(v_tensor)
                        v_emb = sty_fusion(v_emb, v_mask)
                        try:
                            v_tok = v_tok.squeeze(1).cuda()
                            v_tok_len = v_tok_len.cuda()
                            v_sc, v_caps, v_dec_len, _ = decoder(v_emb, v_tok, v_tok_len)
                            v_tgts = v_caps[:, 1:]
                            v_sc_pack = pack_padded_sequence(v_sc, v_dec_len, batch_first=True).data
                            v_tg_pack = pack_padded_sequence(v_tgts, v_dec_len, batch_first=True).data
                            v_loss = criterion(v_sc_pack, v_tg_pack)
                            v_loss_sum += v_loss.item()
                            v_steps += 1
                        except Exception:
                            pass
                cur_val_loss = v_loss_sum / max(1, v_steps)
                print(f"      [Val Loss] Step {index_i} icin Doğrulama Kaybı: {cur_val_loss:.4f}")
                video_encoder.train()
                sty_fusion.train()
                decoder.train()

                with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, index_i, cur_train_loss, cur_val_loss, '', '', '', '', ''])

        # One epoch's validation
        decoder.eval()  # eval mode (no dropout or batchnorm)
        sty_fusion.eval()
        video_encoder.eval()
        val_start_time = time.time()
        references = list()  # references (true captions) for calculating BLEU-4 score
        hypotheses = list()  # hypotheses (predictions)
        
        with torch.no_grad():
            # Batches
            for ind, (video_tensor, token_all, token_all_len, _, _, _,mask) in enumerate(val_loader):
                video_tensor=video_tensor.to(dtype=torch.bfloat16, device='cuda')
                mask=mask.to(dtype=torch.bfloat16, device='cuda')
                vedie_emb,_=video_encoder(video_tensor)
                vedie_emb=sty_fusion(vedie_emb,mask)
                token_all = token_all.squeeze(0).cuda()
                # Forward prop.
                
                seq = decoder.sample(vedie_emb, k=1)
                img_token = token_all.tolist()
                img_tokens = list(map(lambda c: [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}],
                        img_token))  # remove <start> and pads
                references.append(img_tokens)

                pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
                hypotheses.append(pred_seq)
                
                assert len(references) == len(hypotheses)

                if ind % args.print_freq == 0:
                    pred_caption = ""
                    ref_caption = ""
                    for i in pred_seq:
                        pred_caption += (list(word_vocab.keys())[i]) + " "
                    ref_caption = ""
                    for i in img_tokens:
                        for j in i:
                            ref_caption += (list(word_vocab.keys())[j]) + " "
                        ref_caption += ".    "

            val_time = time.time() - val_start_time
            # Calculate evaluation scores
            
            score_dict = get_eval_score(references, hypotheses)
            Bleu_1 = score_dict['Bleu_1']
            Bleu_2 = score_dict['Bleu_2']
            Bleu_3 = score_dict['Bleu_3']
            Bleu_4 = score_dict['Bleu_4']
            Meteor = score_dict['METEOR']
            Rouge = score_dict['ROUGE_L']
            Cider = score_dict['CIDEr']
            print('Validation:\n' 'Time: {0:.3f}\t' 'BLEU-1: {1:.4f}\t' 'BLEU-2: {2:.4f}\t' 'BLEU-3: {3:.4f}\t'
                  'BLEU-4: {4:.4f}\t' 'Rouge: {5:.4f}\t' 'Meteor: {6:.4f}\t', 'Cider: {7:.4f}\t'
                  .format(val_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Rouge, Meteor, Cider))
        
        # Log epoch results to CSV
        with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, index_i, '', '', Bleu_1, Bleu_2, Bleu_3, Bleu_4, Cider])

        #Adjust learning rate
        decoder_lr_scheduler.step()
   
        if  Bleu_4 > best_bleu4:
            best_bleu4 = max(Bleu_4, best_bleu4)
            #save_checkpoint                
            print('Save Model')  
            state = {'video_encoder_dict': video_encoder.state_dict(), 
                    'sty_fusion_dict': sty_fusion.state_dict(),   
                    'decoder_dict': decoder.state_dict(),
                    }                     
            model_name = 'MV_CC'+str(args.data_name)+'_batchsize_'+str(args.train_batchsize)+'_'+str(args.network)+'Bleu_4_'+str(round(10000*Bleu_4))+'.pth'
            torch.save(state, os.path.join(args.savepath, model_name))
            print(os.path.join(args.savepath, model_name))

    print("Eğitim tamamlandı. Eğitim eğrisi (Loss & Metrics) grafikleri oluşturuluyor...")
    try:
        plot_learning_curves(csv_file, args.savepath)
    except Exception as e:
        print(f"Grafik cizim hatasi: {e}")

def plot_learning_curves(csv_path, savepath):
    import pandas as pd
    import matplotlib.pyplot as plt
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    
    # NaN değerleri boş dizeye değil float olarak almak için filtreliyoruz
    df_loss = df.dropna(subset=['Train_Loss', 'Val_Loss'])
    df_metrics = df.dropna(subset=['Bleu_4'])
    
    # 1. Loss Curve
    if len(df_loss) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(df_loss['Step'], df_loss['Train_Loss'], marker='o', markersize=4, color='crimson', label='Train Loss (Aralık: 0 - Sonsuz)')
        plt.plot(df_loss['Step'], df_loss['Val_Loss'], marker='x', markersize=4, color='royalblue', label='Validation Loss (Sık Aralıklarla)')
        plt.title('Training ve Validation Loss Eğrisi (Düştükçe İyidir)')
        plt.xlabel('Eğitim Adımı (Step)')
        plt.ylabel('Kayıp (Loss)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(os.path.join(savepath, 'loss_curve.png'))
        plt.close()

    # 2. Metrics Curve
    if len(df_metrics) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(df_metrics['Epoch'], df_metrics['Bleu_1'].astype(float), marker='^', label='BLEU-1 (Hedef Dağılım: 0.0 - 1.0)')
        plt.plot(df_metrics['Epoch'], df_metrics['Bleu_4'].astype(float), marker='v', label='BLEU-4 (Hedef Dağılım: 0.0 - 1.0)')
        plt.plot(df_metrics['Epoch'], df_metrics['CIDEr'].astype(float), marker='s', label='CIDEr (Hedef Dağılım: 0.0 - ~10.0+)')
        plt.title('Validation Metrik Eğrisi (Epoch Bazlı, Yükseldikçe İyidir)')
        plt.xlabel('Epoch (Tur)')
        plt.ylabel('Skor')
        max_cider = df_metrics['CIDEr'].astype(float).max()
        plt.ylim(0, max(1.1, max_cider + 0.5))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(os.path.join(savepath, 'metrics_curve.png'))
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Changes_to_Captions')

    # Data parameters
    parser.add_argument(
        '--data_folder', default='./Data/LEVIR-MCI-dataset/images', help='folder with data files')
    parser.add_argument('--list_path', default='./data/LEVIR_CC/', help='path of the data lists')
    parser.add_argument('--token_folder', default='./data/LEVIR_CC/tokens/', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=41, help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default="LEVIR_CC",help='base name shared by data files.')
    parser.add_argument('--mode', type=str, default='all', help='mask mode for the dataset (e.g., all, train, val).')

    #parser.add_argument('--data_folder', default='/root/Data/Dubai_CC/DubaiCC500impair/datasetDubaiCCPublic/imgs_tiles/RGB/',help='folder with data files')
    #parser.add_argument('--list_path', default='./data/Dubai_CC/', help='path of the data lists')
    #parser.add_argument('--token_folder', default='./data/Dubai_CC/tokens/', help='folder with token files')
    #parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    #parser.add_argument('--max_length', type=int, default=27, help='path of the data lists')
    #parser.add_argument('--allow_unk', type=int, default=0, help='if unknown token is allowed')
    #parser.add_argument('--data_name', default="Dubai_CC",help='base name shared by data files.')

    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint, None if none.')
    parser.add_argument('--print_freq',type=int, default=100, help='print training/validation stats every __ batches')
    parser.add_argument('--save_steps', type=int, default=5000, help='Save intermediate checkpoint every N steps.')
    # Training parameters
    parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='whether fine-tune encoder or not')    
    parser.add_argument('--train_batchsize', type=int, default=32, help='batch_size for training')
    parser.add_argument('--network', default='resnet101', help='define the encoder to extract features')
    parser.add_argument('--encoder_dim',default=2048, help='the dimension of extracted features using different network')
    parser.add_argument('--feat_size', default=16, help='define the output size of encoder to extract features')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--workers', type=int, default=2, help='for data-loading; right now, only 0 works with h5pys in windows.')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=None, help='clip gradients at an absolute value of.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # Validation
    parser.add_argument('--val_batchsize', type=int, default=1, help='batch_size for validation')
    parser.add_argument('--savepath', default="./models_checkpoint/")
    # Model parameters
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--attention_dim', type=int, default=2048)
    parser.add_argument('--feature_dim', type=int, default=2048)
    args = parser.parse_args()
    main(args)
