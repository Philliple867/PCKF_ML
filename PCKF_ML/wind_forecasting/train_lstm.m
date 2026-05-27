addpath(genpath('../core'));
cfg = config();
rng(cfg.seed);

v = ver('nnet');
if isempty(v); error('Deep Learning Toolbox not installed.'); end
fprintf('Deep Learning Toolbox v%s\n\n', v.Version);

tr  = readtable(cfg.f_train,'Delimiter',',','VariableNamingRule','preserve');
va  = readtable(cfg.f_val,  'Delimiter',',','VariableNamingRule','preserve');
Xtr = table2array(tr(:,cfg.features))';
Xva = table2array(va(:,cfg.features))';
ytr = tr.(cfg.target)';
yva = va.(cfg.target)';

mu_X=mean(Xtr,2); sd_X=std(Xtr,0,2); sd_X(sd_X<1e-8)=1;
mu_y=mean(ytr);   sd_y=std(ytr);

Xtr_n=(Xtr-mu_X)./sd_X; ytr_n=(ytr-mu_y)./sd_y;
Xva_n=(Xva-mu_X)./sd_X; yva_n=(yva-mu_y)./sd_y;

[Xseq_tr,yseq_tr]=sliding_window(Xtr_n,ytr_n,cfg.seq_len);
[Xseq_va,yseq_va]=sliding_window(Xva_n,yva_n,cfg.seq_len);
ntr=numel(Xseq_tr); nva=numel(Xseq_va); nf=size(Xtr,1);
fprintf('Train: %d seqs | Val: %d seqs | Features: %d\n',ntr,nva,nf);

layers=[
    sequenceInputLayer(nf)
    lstmLayer(cfg.hidden1,'OutputMode','last')
    dropoutLayer(cfg.dropout)
    lstmLayer(cfg.hidden2,'OutputMode','last')
    dropoutLayer(cfg.dropout)
    fullyConnectedLayer(2)];
net=dlnetwork(layerGraph(layers));
np=sum(cellfun(@(x) numel(extractdata(x)), net.Learnables.Value));
fprintf('Parameters: %d\n\n',np);

lr=cfg.lr0; avg_g=[]; avg_sg=[]; iter=0;
best_val=Inf; no_imp=0;
fprintf('%-6s  %-12s  %-12s\n','epoch','train_nll','val_nll');

for ep=1:cfg.epochs
    perm=randperm(ntr); nb=floor(ntr/cfg.batchsz); ep_loss=0;

    for b=1:nb
        iter=iter+1;
        idx=perm((b-1)*cfg.batchsz+1:b*cfg.batchsz);

        % use 'CTB' instead of cell array 'CT'
        Xb=dlarray(single(cat(3,Xseq_tr{idx})),'CTB');
        yb=dlarray(single(yseq_tr(idx)'),'CB');

        [loss_b,grad_b]=dlfeval(...
            @(n,X,y) deal(nll_loss(n,X,y,cfg.lsig_min,cfg.lsig_max),...
                dlgradient(nll_loss(n,X,y,cfg.lsig_min,cfg.lsig_max),n.Learnables)),...
            net,Xb,yb);

        % gradient clipping
        gvals=grad_b.Value; gn=0;
        for k=1:numel(gvals)
            g=extractdata(gvals{k}); gn=gn+sum(g(:).^2);
        end
        if sqrt(gn)>cfg.gradclip
            grad_b=dlupdate(@(g) g*(cfg.gradclip/sqrt(gn)),grad_b);
        end

        [net,avg_g,avg_sg]=adamupdate(net,grad_b,avg_g,avg_sg,iter,lr);
        ep_loss=ep_loss+extractdata(loss_b);
    end
    tnll=ep_loss/nb;

    % validation
    vnll=0; nc=0; bs_v=512;
    for b=1:ceil(nva/bs_v)
        i1=(b-1)*bs_v+1; i2=min(b*bs_v,nva); if i1>nva; break; end
        Xvb=dlarray(single(cat(3,Xseq_va{i1:i2})),'CTB');
        yvb=dlarray(single(yseq_va(i1:i2)'),'CB');
        vl=dlfeval(@nll_loss,net,Xvb,yvb,cfg.lsig_min,cfg.lsig_max);
        vnll=vnll+extractdata(vl)*(i2-i1+1); nc=nc+(i2-i1+1);
    end
    vnll=vnll/nc;

    if vnll<best_val; best_val=vnll; best_net=net; no_imp=0; tag=' <--';
    else; no_imp=no_imp+1; tag=''; end
    if no_imp>=cfg.patience
        lr=lr*cfg.lr_decay; no_imp=0;
        fprintf('  [lr -> %.2e at epoch %d]\n',lr,ep);
    end
    if mod(ep,5)==0||ep==1
        fprintf('%-6d  %-12.4f  %-12.4f%s\n',ep,tnll,vnll,tag);
    end
end
fprintf('\nBest val NLL: %.4f\n',best_val);
if ~isfolder(cfg.res_dir); mkdir(cfg.res_dir); end
save(cfg.f_model,'best_net','mu_X','sd_X','mu_y','sd_y');
fprintf('Saved: %s\n',cfg.f_model);