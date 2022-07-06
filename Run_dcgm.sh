chmod +x allworkload.sh
chmod +x dcgmi_field1.sh
chmod +x dcgmi_field2.sh
chmod +x dcgmi_field3.sh

nohup ./allworkload.sh >nohup.out & nohup ./dcgmi_field1.sh >nohup2.out & nohup ./dcgmi_field2.sh >nohup3.out & nohup ./dcgmi_field3.sh >nohup4.out &
