# python test_single.py --ckpt checkpoint/left_eye_256/cyclegan.pb --input datasets/left_eye/testCustom/boy.jpeg --output rjq.jpg

dir=datasets/left_eye/testA

for i in $dir/*
do
    echo $i
    python test_single.py --ckpt checkpoint/left_eye_256/cyclegan.pb --input $i --output testresult
done