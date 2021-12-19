<아래 명령어 2개 동시에 돌림>
- python3 sample.py 
- nvidia-smi -l 1 -f ./Data.csv --format=csv --query-gpu=timestamp,driver_version,count,name,serial,uuid,pci.bus_id,pci.domain,pci.bus,pci.device,pci.device_id,pci.sub_device_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,index,display_mode,display_active,persistence_mode,accounting.mode,accounting.buffer_size,driver_model.current,driver_model.pending,vbios_version,inforom.img,inforom.oem,inforom.ecc,inforom.pwr,gom.current,gom.pending,fan.speed,pstate,clocks_throttle_reasons.supported,clocks_throttle_reasons.active,clocks_throttle_reasons.gpu_idle,clocks_throttle_reasons.applications_clocks_setting,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_slowdown,memory.total,memory.used,memory.free,compute_mode,utilization.gpu,utilization.memory,ecc.mode.current,ecc.mode.pending,ecc.errors.corrected.volatile.device_memory,ecc.errors.corrected.volatile.register_file,ecc.errors.corrected.volatile.l1_cache,ecc.errors.corrected.volatile.l2_cache,ecc.errors.corrected.volatile.texture_memory,ecc.errors.corrected.volatile.total,ecc.errors.corrected.aggregate.device_memory,ecc.errors.corrected.aggregate.register_file,ecc.errors.corrected.aggregate.l1_cache,ecc.errors.corrected.aggregate.l2_cache,ecc.errors.corrected.aggregate.texture_memory,ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.volatile.device_memory,ecc.errors.uncorrected.volatile.register_file,ecc.errors.uncorrected.volatile.l1_cache,ecc.errors.uncorrected.volatile.l2_cache,ecc.errors.uncorrected.volatile.texture_memory,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.device_memory,ecc.errors.uncorrected.aggregate.register_file,ecc.errors.uncorrected.aggregate.l1_cache,ecc.errors.uncorrected.aggregate.l2_cache,ecc.errors.uncorrected.aggregate.texture_memory,ecc.errors.uncorrected.aggregate.total,retired_pages.single_bit_ecc.count,retired_pages.double_bit.count,retired_pages.pending,temperature.gpu,power.management,power.draw,power.limit,power.default_limit,power.min_limit,power.max_limit,clocks.current.graphics,clocks.current.sm,clocks.current.memory,clocks.applications.graphics,clocks.applications.memory,clocks.default_applications.graphics,clocks.default_applications.memory,clocks.max.graphics,clocks.max.sm,clocks.max.memory

파이썬 파일안에서 에포크 한바퀴 돌때마다 nvidia-smi -q 값들을 모아서 데이터로 저장

<저장한 데이터는 다음과같이 로컬에서 저장가능>

(맥)
scp -i /Users/heoyunseo/desktop/aws_pem/ys.pem [ubuntu@](public ip 주소):/home/ubuntu/flee/Test/(파일명 epoch1.csv) .
폴더째 정송할때 : 
scp -i /Users/heoyunseo/desktop/aws_pem/ys.pem -r ubuntu@[public ip 주소]:/home/ubuntu/Hardware-Data/ .

(윈도우)
scp -i C:/Users/Owner/Desktop/ys2.pem -r ubuntu@54.226.138.113:/home/ubuntu/Hardware-Data/ .
