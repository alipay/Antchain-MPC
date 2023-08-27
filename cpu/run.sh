source ./stf_venv/bin/activate
export MAIN_PATH=${PWD}
export PYTHONPATH=$PYTHONPATH:$MAIN_PATH
cd examples


echo "------------------run test for Table 8-------------------------"
echo "run test for Table8_AlexNet"
python3 run_AlexNet.py --epoch=0.01  --predict_flag=0 --batch_size=32 --config_file="../conf/config_linear.json" >$MAIN_PATH/artifacts/Table8_AlexNet_32.log 2>&1
python3 run_AlexNet.py --epoch=0.25  --predict_flag=0 --batch_size=128 --config_file="../conf/config_linear.json" >$MAIN_PATH/artifacts/Table8_AlexNet_128.log 2>&1


echo "run test for Table8_VGG16"
python3 run_VGG16.py --epoch=0.01  --predict_flag=0 --batch_size=32 --config_file="../conf/config_linear.json" >$MAIN_PATH/artifacts/Table8_VGG16_32.log 2>&1


echo "run test for Table8_AlexNet_TinyImageNet"
python3 run_AlexNet_ti.py --epoch=0.25  --predict_flag=0 --batch_size=128 --config_file="../conf/config_linear.json" >$MAIN_PATH/artifacts/Table8_AlexNet_Ti_128.log 2>&1


echo "run test for Table8_VGG16_TinyImageNet"
python3 run_vgg16_ti.py --epoch=0.1  --predict_flag=0 --batch_size=8 --config_file="../conf/config_linear.json" >$MAIN_PATH/artifacts/Table8_VGG16_Ti_8.log 2>&1
python3 run_vgg16_ti.py --epoch=0.1  --predict_flag=0 --batch_size=32 --config_file="../conf/config_linear.json" >$MAIN_PATH/artifacts/Table8_VGG16_Ti_32.log 2>&1










echo "----------------run test for  Table9,11-------------------------------"
# 跑 Table 11时候需要限制网络带宽和时延。 
# 其中 Table 11 的 5ms对应了
# tc qdisc del dev lo root
# DELAY_MS=2.5
# RATE_MBIT=100
# tc qdisc replace dev lo root netem delay ${DELAY_MS}ms rate ${RATE_MBIT}Mbit
# 其中 Table 11 的 50ms对应了
# tc qdisc del dev lo root
# DELAY_MS=25
# RATE_MBIT=100
# tc qdisc replace dev lo root netem delay ${DELAY_MS}ms rate ${RATE_MBIT}Mbit

echo "run test for Table9,11:Ours-T_networkA"
python3 run_networkA.py --epoch=0.1  --predict_flag=0  >$MAIN_PATH/artifacts/Table9_networkA_T.log 2>&1


echo "run test for Table9,11:Ours-T_networkB"
python3 run_networkB.py --epoch=0.1  --predict_flag=0  >$MAIN_PATH/artifacts/Table9_networkB_T.log 2>&1


echo "run test for Table9,11:Ours-T_networkC"
python3 run_networkC.py --epoch=0.1  --predict_flag=0  >$MAIN_PATH/artifacts/Table9_networkC_t.log 2>&1


echo "run test for Table9,11:Ours-T_networkD"
python3 run_networkD.py --epoch=0.1  --predict_flag=0  >$MAIN_PATH/artifacts/Table9_networkD_T.log 2>&1



echo "run test for Table11:Ours-B_networkA"
python3 run_networkA.py --epoch=0.1  --predict_flag=0 --config_file="../conf/config_linear.json"  >$MAIN_PATH/artifacts/Table11_networkA_B.log 2>&1


echo "run test for Table11:Ours-B_networkB"
python3 run_networkB.py --epoch=0.1  --predict_flag=0 --config_file="../conf/config_linear.json"  >$MAIN_PATH/artifacts/Table11_networkB_B.log 2>&1


echo "run test for Table11:Ours-B_networkC"
python3 run_networkC.py --epoch=0.1  --predict_flag=0 --config_file="../conf/config_linear.json"  >$MAIN_PATH/artifacts/Table11_networkC_B.log 2>&1


echo "run test for Table11:Ours-B_networkD"
python3 run_networkD.py --epoch=0.1  --predict_flag=0 --config_file="../conf/config_linear.json" >$MAIN_PATH/artifacts/Table11_networkD_B.log 2>&1










echo "----------------run test for  Table12-------------------------------"
tc qdisc del dev lo root
DELAY_MS=30
RATE_MBIT=320
tc qdisc replace dev lo root netem delay ${DELAY_MS}ms rate ${RATE_MBIT}Mbit
python3 run_networkC.py --epoch=0.1  --predict_flag=0  >$MAIN_PATH/artifacts/Table12_LeNet_T.log 2>&1
python3 run_networkC.py --epoch=0.1  --predict_flag=0 --config_file="../conf/config_linear.json"  >$MAIN_PATH/artifacts/Table12_LeNet_B.log 2>&1
python3 run_AlexNet.py --epoch=0.01  --predict_flag=0 --batch_size=128 >$MAIN_PATH/artifacts/Table12_AlexNet_T.log 2>&1
python3 run_AlexNet.py --epoch=0.01  --predict_flag=0 --batch_size=128 --config_file="../conf/config_linear.json" >$MAIN_PATH/artifacts/Table12_AlexNet_B.log 2>&1
tc qdisc del dev lo root











echo "------------------run test for Table 7-------------------------"
echo "run test for Table7_networkA"
python3 run_networkA.py --epoch=1  --pretrain=0 > $MAIN_PATH/artifacts/Table7_networkA_epoch1_pretrain0.log 2>&1
python3 run_networkA.py --epoch=1  --pretrain=1 > $MAIN_PATH/artifacts/Table7_networkA_epoch1_pretrain1.log 2>&1
python3 run_networkA.py --epoch=5  --pretrain=0 > $MAIN_PATH/artifacts/Table7_networkA_epoch5_pretrain0.log 2>&1
python3 run_networkA.py --epoch=5  --pretrain=1 > $MAIN_PATH/artifacts/Table7_networkA_epoch5_pretrain1.log 2>&1


echo "run test for Table_networkB"
python3 run_networkB.py --epoch=1  --pretrain=0 --pooling=max > $MAIN_PATH/artifacts/Table7_networkB_epoch1_pretrain0.log 2>&1
python3 run_networkB.py --epoch=1  --pretrain=1 --pooling=max > $MAIN_PATH/artifacts/Table7_networkB_epoch1_pretrain1.log 2>&1
python3 run_networkB.py --epoch=5  --pretrain=0 --pooling=max > $MAIN_PATH/artifacts/Table7_networkB_epoch5_pretrain0.log 2>&1
python3 run_networkB.py --epoch=5  --pretrain=1 --pooling=max > $MAIN_PATH/artifacts/Table7_networkB_epoch5_pretrain1.log 2>&1


echo "run test for Table7_networkC (LeNet)"
python3 run_networkC.py --epoch=1  --pretrain=0 --pooling=max > $MAIN_PATH/artifacts/Table7_networkC_epoch1_pretrain0.log 2>&1
python3 run_networkC.py --epoch=1  --pretrain=1 --pooling=max > $MAIN_PATH/artifacts/Table7_networkC_epoch1_pretrain1.log 2>&1
python3 run_networkC.py --epoch=5  --pretrain=0 --pooling=max > $MAIN_PATH/artifacts/Table7_networkC_epoch5_pretrain0.log 2>&1
python3 run_networkC.py --epoch=5  --pretrain=1 --pooling=max > $MAIN_PATH/artifacts/Table7_networkC_epoch5_pretrain1.log 2>&1


echo "run test for Table7_networkD"
python3 run_networkD.py --epoch=1  --pretrain=0  > $MAIN_PATH/artifacts/Table7_networkD_epoch1_pretrain0.log 2>&1
python3 run_networkD.py --epoch=1  --pretrain=1  > $MAIN_PATH/artifacts/Table7_networkD_epoch1_pretrain1.log 2>&1
python3 run_networkD.py --epoch=5  --pretrain=0  > $MAIN_PATH/artifacts/Table7_networkD_epoch5_pretrain0.log 2>&1
python3 run_networkD.py --epoch=5  --pretrain=1  > $MAIN_PATH/artifacts/Table7_networkD_epoch5_pretrain1.log 2>&1


echo "run test for Table7_ALexNet (slowly)"

python3 run_AlexNet.py --epoch=5  --pretrain=1 --truncation_type=1 > $MAIN_PATH/artifacts/Table7_Alexnet_epoch5_pretrain1.log 2>&1
python3 run_AlexNet.py --epoch=10  --pretrain=1 --truncation_type=1 > $MAIN_PATH/artifacts/Table7_Alexnet_epoch10_pretrain1.log 2>&1
python3 run_AlexNet.py --epoch=5  --pretrain=0  --truncation_type=1 > $MAIN_PATH/artifacts/Table7_Alexnet_epoch5_pretrain0.log 2>&1
python3 run_AlexNet.py --epoch=10  --pretrain=0  --truncation_type=1 > $MAIN_PATH/artifacts/Table7_Alexnet_epoch10_pretrain0.log 2>&1







echo "run test for Table7_VGG16  (very slowly)"

python3 run_VGG16.py --epoch=5  --pretrain=1  --pooling max --truncation_type=1 > $MAIN_PATH/artifacts/Table7_VGG16_epoch5_pretrain1.log 2>&1
python3 run_VGG16.py --epoch=10  --pretrain=1 --pooling max  --truncation_type=1 > $MAIN_PATH/artifacts/Table7_VGG16_epoch10_pretrain1.log 2>&1
python3 run_VGG16.py --epoch=5  --pretrain=0  --pooling max --truncation_type=1 > $MAIN_PATH/artifacts/Table7_VGG16_epoch5_pretrain0.log 2>&1
python3 run_VGG16.py --epoch=10  --pretrain=0 --pooling max --truncation_type=1 > $MAIN_PATH/artifacts/Table7_VGG16_epoch10_pretrain0.log 2>&1



