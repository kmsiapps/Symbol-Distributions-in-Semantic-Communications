experiment=SemELIC_small_10dB_kl_L100
echo $experiment
echo `cat logs/$experiment/nohup.out | tail -2 | head -1`
echo `cat logs/$experiment/nohup.out | tail -1`
echo ""

experiment=SemELIC_small_10dB_kl_L1000
echo $experiment
echo `cat logs/$experiment/nohup.out | tail -2 | head -1`
echo `cat logs/$experiment/nohup.out | tail -2`
echo ""

experiment=SemELIC_small_10dB_kl_L10000
echo $experiment
echo `cat logs/$experiment/nohup.out | tail -2 | head -1`
echo `cat logs/$experiment/nohup.out | tail -2`

