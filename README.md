# circuit-scribble-server
SignalR ONNX/pytorch Service wit Azure Xilinx u250 acceleration


# CircuitScribble

CircuitScribble is a SignalR Service for u250 on Azure NP10 VM

Two Versions

1. Pytorch "Hello World" version will train a small training set and launch a service to serve that trained set
2. ONNX version will load a pre-trained onnx model and clone

STATUS: Pending a Vitus 2.0
1. The 1.4 Vitis compiler seems to result in low accuracy when run on hardware
2. The ONNX Vitis runtime is based on the dpuv1 and is not compatible with the current dpu-azure.



# Vitis-AI on Microsoft Azure
The following steps will guide you on how to build up a VM capable of running Vitis-AI

## Steps

- [Start an Azure VM](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/quick-create-portal) of type [NP10](https://docs.microsoft.com/en-us/azure/virtual-machines/np-series)  using the Cannonical [Ubuntu 18.04 LTS](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/canonical.0001-com-ubuntu-server-bionic?tab=overview) app
```
After starting this instance you must ssh to your cloud instance to complete the following steps
```

- Disable Kernel Auto-Upgrade
```
sudo sed -i 's/1/0/g' /etc/apt/apt.conf.d/20auto-upgrades
```

- Update Ubuntu Packages List, and Upgrade existing packages
```
sudo apt-get update && sudo apt-get upgrade
```

- Install Xilinx XRT
```
git clone https://github.com/Xilinx/XRT.git
cd XRT 
git checkout 2021.1
sudo ./src/runtime_src/tools/scripts/xrtdeps.sh
cd build
./build.sh
sudo apt install ./Release/*-xrt.deb
sudo apt install ./Release/*-azure.deb
cd

# XCLMGMT Driver needs to be removed within the VM
sudo modprobe -r xclmgmt
sudo reboot

# At this point, re-establish your ssh connection
ssh ...
```

- Install XRM
```
wget https://www.xilinx.com/bin/public/openDownload?filename=xrm_202110.1.2.1539_18.04-x86_64.deb -O xrm.deb
sudo apt install ./xrm.deb
```

- Install the DPU Accelerator (FPGA Binary)
```
wget https://www.xilinx.com/bin/public/openDownload?filename=dpu-azure-1.4.0.xclbin -O dpu-azure.xclbin
sudo mkdir -p /opt/xilinx/overlaybins/DPUCADF8H
sudo cp dpu-azure.xclbin /opt/xilinx/overlaybins/DPUCADF8H
sudo chmod -R a+rx /opt/xilinx/overlaybins/DPUCADF8H
```

- [Install Docker](https://docs.docker.com/engine/install/ubuntu/)
```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo usermod -aG docker ubuntu
# At this point, re-establish your ssh connection, so the ubuntu user can run docker commands
exit
ssh ...
```

- Pull a known comptabile pre-build docker vitis container
```
docker pull xilinx/vitis-ai:1.4.1.978
```
- Check out this repo 
https://github.com/virtualbreadboard/circuit-scribble-server.git


.. standby run examples.

* vbb_signalr_py_server - Reference functional example 

 