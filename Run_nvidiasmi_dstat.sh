chmod +x dstat.sh
chmod +x nvidiasmi.sh
chmod +x allworkload.sh
nohup ./dstat.sh >nohup.out & nohup ./nvidiasmi.sh >nohup2.out & nohup ./allworkload.sh >nohup3.out & 
