pip3 install tensorflow==2.7.0

sudo rm /var/lib/apt/lists/lock
sudo rm /var/cache/apt/archives/lock
sudo rm /var/lib/dpkg/lock*
sudo dpkg --configure -a 
sudo apt update
sudo apt install dstat

chmod +x run_all.sh
