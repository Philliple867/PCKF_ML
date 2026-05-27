% R2025a 
addpath(genpath('../core'));
cfg = config();
rng(cfg.seed);

if ~isfile(cfg.f_model); error('Run train_lstm.m first.'); end
load(cfg.f_model,'best_net','mu_X','sd_X','mu_y','sd_y');

te    = readtable(cfg.f_test,'Delimiter',',','VariableNamingRule','preserve');
Xte   = table2array(te(:,cfg.features))';
yte   = te.(cfg.target)';
Xte_n = (Xte-mu_X)./sd_X;
yte_n = (yte-mu_y)./sd_y;

[Xseq,yseq]=sliding_window(Xte_n,yte_n,cfg.seq_len);
N=numel(Xseq);

N_mc=30; mu_mc=zeros(N_mc,N,'single'); sig_mc=zeros(N_mc,N,'single');
fprintf('MC Dropout (%d passes)...\n',N_mc);
bs=256;

for m=1:N_mc
    m_all=zeros(1,N,'single'); s_all=zeros(1,N,'single');
    for b=1:ceil(N/bs)
        i1=(b-1)*bs+1; i2=min(b*bs,N);
        XB=dlarray(single(cat(3,Xseq{i1:i2})),'CTB');
        pr=extractdata(predict(best_net,XB));
        m_all(i1:i2)=pr(1,:);
        ls=max(min(pr(2,:),cfg.lsig_max),cfg.lsig_min);
        s_all(i1:i2)=exp(ls);
    end
    mu_mc(m,:)=m_all; sig_mc(m,:)=s_all;
end

mu_ens  = mean(mu_mc,1)';
sig_ens = sqrt(mean(sig_mc.^2,1)'+var(mu_mc,0,1)');
mu_pred  = mu_ens*sd_y+mu_y;
sig_pred = max(sig_ens*sd_y,0.05);
true_mps = yseq*sd_y+mu_y;

rmse_v   = sqrt(mean((mu_pred-true_mps).^2));
mae_v    = mean(abs(mu_pred-true_mps));
z90      = 1.6449;
coverage = 100*mean(true_mps>=mu_pred-z90*sig_pred & true_mps<=mu_pred+z90*sig_pred);

fprintf('\nTest 2019:\n');
fprintf('  RMSE     = %.4f m/s\n',rmse_v);
fprintf('  MAE      = %.4f m/s\n',mae_v);
fprintf('  90%% PI   = %.1f%%\n', coverage);
fprintf('  mu sigma = %.4f m/s  range [%.4f, %.4f]\n',...
    mean(sig_pred),min(sig_pred),max(sig_pred));
if coverage<85
    warning('Coverage %.1f%% below 85%%.', coverage);
end

time_col=te.Time(cfg.seq_len+1:end);
T_out=table(time_col,mu_pred,sig_pred,true_mps,...
    'VariableNames',{'Time','mu_wind','sigma_wind','true_wind'});
if ~isfolder(cfg.res_dir); mkdir(cfg.res_dir); end
writetable(T_out,cfg.f_lstm);
fprintf('\nSaved: %s  (%d rows)\n',cfg.f_lstm,height(T_out));