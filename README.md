nvidia-smi , dstat 을 수집함 / dcgmi 는 전체 피처 다 뽑은다음 솎아내기용으로 사용

- aws configure 로 리전 맞춰주고

cuda11.4 /
tensorflow 2.7.0 /
python 3.7 /
Deep Learning AMI GPU CUDA 11.4.1 (Ubuntu 18.04) 20211204 AMI)

- 로컬에서 파일 다운로드받고, 파일안에 pem 키넣고
- sh startCLI_Oregon.sh g4dn.xlarge

다깔고 ssh 접속되면
- cd Hardware-Data-Collect
- sudo bash ./Run_nvidiasmi_dstat.sh (gpu+cpu둘다수집)
- sudo bash ./Run_nvidiasmi.sh (gpu만수집)
- sudo bash ./Run_dcgm.sh (dcgmi로 gpu수집)

로컬로 다운로드
- scp -i /Users/yoonseo/desktop/aws_pem/ys-oregon2.pem -r ubuntu@54.185.254.117:/home/ubuntu/Hardware-Data-Collect/ .
