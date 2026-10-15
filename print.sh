
flag=0
while [ $flag -eq 0 ]
    do
        clear
        
        experiment=fSemELIC_sm10dB_base_fac
        echo $experiment
        echo `cat logs/$experiment/nohup.out | tail -2 | head -1`
        echo `cat logs/$experiment/nohup.out | tail -1`
        echo ""
        experiment=fSemELIC_sm10dB_KLdv
        echo $experiment
        echo `cat logs/$experiment/nohup.out | tail -2 | head -1`
        echo `cat logs/$experiment/nohup.out | tail -1`
        echo ""
        experiment=fSemELIC_sm10dB_KLdv_fac
        echo $experiment
        echo `cat logs/$experiment/nohup.out | tail -2 | head -1`
        echo `cat logs/$experiment/nohup.out | tail -1`
        echo ""
        experiment=fSemELIC_sm10dB_KLdv_fac_B8
        echo $experiment
        echo `cat logs/$experiment/nohup.out | tail -2 | head -1`
        echo `cat logs/$experiment/nohup.out | tail -1`
        echo ""

        sleep 1
    done