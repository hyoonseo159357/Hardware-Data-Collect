pip3 install tensorflow==2.7.0

sudo apt update
sudo killall apt apt-get
sudo rm /var/lib/apt/lists/lock
sudo rm /var/cache/apt/archives/lock
sudo rm /var/lib/dpkg/lock*
sudo dpkg --configure -a
sudo apt update

sudo apt install dstat

chmod +x run_all.sh

chmod +x run_dstat.sh
chmod +x run_nvidia.sh
chmod +x run_workload.sh
nohup ./run_dstat.sh >nohup.out & nohup ./run_nvidia.sh >nohup2.out & nohup ./run_workload.sh >nohup3.out & 
