experiment=SemELIC_sm10dB_KLdv_fac_Kp5e3
echo $experiment
echo `cat logs/$experiment/nohup.out | tail -2 | head -1`
echo `cat logs/$experiment/nohup.out | tail -1`
echo ""

experiment=NTSCC_Lo18_KLdv_K1
echo $experiment
# echo `cat logs/$experiment/logger.log | tail -2 | head -1`
echo `cat logs/$experiment/logger.log | tail -1`
echo ""

experiment=NTSCC_Lo18_KLdv_Kpe2
echo $experiment
# echo `cat logs/$experiment/logger.log | tail -2 | head -1`
echo `cat logs/$experiment/logger.log | tail -1`

