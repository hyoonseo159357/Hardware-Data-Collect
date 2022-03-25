cuda11.4
tensorflow 2.7.0

- 로컬에서 파일 다운로드받고, 파일안에 pem 키넣고
- sh startCLI.sh g4dn.xlarge

다깔고 ssh 접속되면
- cd Hardware-Data2
- sudo bash ./run_all.sh (gpu+cpu둘다수집)
- sudo bash ./run_nvidiasmi.sh (gpu만수집)


