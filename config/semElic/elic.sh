
# config/semElic/elic.sh

# Baseline SemElic
dh push --exp SemELIC_small_10dB_log python train_sem.py config/semElic/config_semelic_small_log.yaml

# SemElic with KL Divergence loss
experiment=SemELIC_small_10dB_kl_L100
mkdir logs/$experiment
nohup python train_sem.py config/semElic/config_semelic_small_kl100.yaml 1>logs/$experiment/nohup.out 2>&1 &
experiment=SemELIC_small_10dB_kl_L1000
mkdir logs/$experiment
nohup python train_sem.py config/semElic/config_semelic_small_kl1000.yaml 1>logs/$experiment/nohup.out 2>&1 &
experiment=SemELIC_small_10dB_kl_L10000
mkdir logs/$experiment
nohup python train_sem.py config/semElic/config_semelic_small_kl10000.yaml 1>logs/$experiment/nohup.out 2>&1 &
dh push --exp SemELIC_small_10dB_kl_L10000 python train_sem.py config/semElic/config_semelic_small_kl10000_resume.yaml
dh push --exp SemELIC_small_10dB_kl_L1e-5 python train_sem.py config/semElic/config_semelic_small_kl1e-5.yaml
dh push --exp SemELIC_small_10dB_kl_L1e-6 python train_sem.py config/semElic/config_semelic_small_kl1e-6.yaml

# SemElic with variance h (kernel size)
dh push --exp SemELIC_small_10dB_kl_L1e-4_h1e-1 python train_sem.py config/semElic/config_semelic_small_kl1e-4_h1e-1.yaml
dh push --exp SemELIC_small_10dB_kl_L1e-4_h5e-1 python train_sem.py config/semElic/config_semelic_small_kl1e-4_h5e-1.yaml

# Decaying lambda
dh push --exp SemELIC_small_10dB_kl_L1e-1_decay python train_sem.py config/semElic/config_semelic_small_kl1e-1_decay.yaml

# Reduced lambda
dh push --exp SemELIC_small_10dB_kl_L1e-5_redu python train_sem.py config/semElic/config_semelic_small_kl1e-5_redu.yaml

# 20dB models
dh push --exp SemELIC_small_20dB_log python train_sem.py config/semElic/config_semelic20_small_log.yaml
dh push --exp SemELIC_small_20dB_kl_L1e-4 python train_sem.py config/semElic/config_semelic20_small_kl1e-4.yaml
dh push --exp SemELIC_small_20dB_kl_L1e-5 python train_sem.py config/semElic/config_semelic20_small_kl1e-5.yaml
dh push --exp SemELIC_small_20dB_nkl_L1e-4 python train_sem.py config/semElic/config_semelic20_small_nkl1e-4.yaml