## Finetune Qwen2-VL-7B wtih LLaMA Factory

This repository demonstrates the fine-tuning process of the multi-modal Qwen2-VL-7B model using Amazon SageMaker Hyperpod. It provides a comprehensive guide and code examples for leveraging the powerful Hyperpod infrastructure to efficiently fine-tune the Qwen2-VL-7B model, which combines vision and language capabilities. The repository includes Python scripts for the fine-tuning process and Slurm configurations for distributed training, enabling users to scale their workloads across multiple nodes. By following this guide, data scientists and machine learning engineers can harness the full potential of Qwen2-VL-7B for various multi-modal tasks while taking advantage of SageMaker Hyperpod's scalable and cost-effective distributed training capabilities. The provided scripts and configurations streamline the fine-tuning workflow, allowing users to optimize the model for their specific use cases.


## Cluster Creation

You can follow the AWS workshop content with step by step guidance. 
https://catalog.workshops.aws/sagemaker-hyperpod/en-US

### Lifecycle Scripts

Lifecycle scripts allow customization of your cluster during creation. They will be used to install software packages. The official lifecycle scripts is suitable for general use-cases. 

To set up lifecycle scripts:

1. Clone the repository and upload scripts to S3:
   ```bash
   git clone --depth=1 https://github.com/aws-samples/awsome-distributed-training/
   cd awsome-distributed-training/1.architectures/5.sagemaker-hyperpod/LifecycleScripts/
   aws s3 cp --recursive base-config/ s3://${BUCKET}/src
   ```

### Cluster Configuration

1. Prepare `cluster-config.json` and `provisioning_parameters.json` files.
2. Upload the configuration to S3:
   ```bash
   aws s3 cp provisioning_parameters.json s3://${BUCKET}/src/
   ```
3. Create the cluster:
   ```bash
   aws sagemaker create-cluster --cli-input-json file://cluster-config.json --region $AWS_REGION
   ```

Example of [`cluster-config.json` and `provisioning_parameters.json` can be found at  in ClusterConfig](./cluster_config)

 
### Scaling the Cluster

To increase worker instances:

1. Update `cluster-config.json` with the new instance count.
2. Run:
   ```bash
   aws sagemaker update-cluster \
    --cluster-name ${my-cluster-name} \
    --instance-groups file://update-cluster-config.json \
    --region $AWS_REGION
   ```

### Shutting Down the Cluster

```bash
aws sagemaker delete-cluster --cluster-name ${my-cluster-name}
```

### Notes

- SageMaker HyperPod supports Amazon FSx for Lustre integration, enabling [full bi-directional synchronization with Amazon S3](https://aws.amazon.com/blogs/aws/enhanced-amazon-s3-integration-for-amazon-fsx-for-lustre/).
- Ensure proper AWS CLI permissions and configurations. 
- Validate the cluster configuration files before lauching the cluster
```
curl -O https://raw.githubusercontent.com/aws-samples/awsome-distributed-training/main/1.architectures/5.sagemaker-hyperpod/validate-config.py

pip3 install boto3
python3 validate-config.py --cluster-config cluster-config.json --provisioning-parameters provisioning_parameters.json
```

## Cluster connection

### SSH into controller node 
If you are using SageMaker HyperPod, you might follow the tutorial here to setup up SSH connection.

SSH into cluster 
```
./easy-ssh.sh -c controller-machine ml-cluster
sudo su - ubuntu
```
### Connect with VSCode on local machine  
SageMaker HyperPod supports connecting to the cluster via VSCode. You can setup a SSH Proxy via SSM and use that to connect in Visual Studio Code, following this guidance
https://catalog.workshops.aws/sagemaker-hyperpod/en-US/05-advanced/05-vs-code

## Training and evaluation

All the following steps will be executed on GPUs nodes i.e. 2 * g5.2xlarge, you can ssh into worker node https://catalog.workshops.aws/sagemaker-hyperpod/en-US/01-cluster/07-ssh-compute 

```
sinfo 
ssh ip-10-1-23-***
```


### Miniconda install 
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -f -p ~/miniconda3
source ~/miniconda3/bin/activate
```

### Create environment on worker node (i.e. g5.2xlarge)
```
conda create -n llamafactory python=3.10
conda activate llamafactory
```
```
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[torch,metrics,deepspeed,bitsandbytes,liger-kernel]" "transformers>=0.45.0"
pip install flash-attn
cd ..
```

### Start the data pre-processing
clone the current repository and cd into repo 
```
git clone https://github.com/aws-samples/fine-tune-qwen2-vl-with-llama-factory.git
cd fine-tune-qwen2-vl-with-llama-factory
#pip install -r requirements.txt
python ./preprocessing/process_fintabnet_en.py --output_dir ./data/fintabnet_en
```

Add pubtabnet format in `./data/dataset_info.json` (added fintabnet_en with this sample code)

### Prepare PiSSA Qwen2-VL Model
PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models. PiSSA shares the same architecture as LoRA. However compared to LoRA, PiSSA updates the principal components while freezing the "residual" parts, allowing faster convergence and enhanced performance.

```bash
python ./train_configs/pissa_init.py --model_name_or_path Qwen/Qwen2-VL-7B-Instruct --output_dir models/qwen2_vl_7b_pissa_128 --lora_rank 128 --lora_target $'^(?!.*patch_embed).*(?:gate_proj|k_proj|fc2|o_proj|v_proj|up_proj|fc1|proj|down_proj|qkv|q_proj).*'
```

### Start the training job

Prepare training config `./train_configs/train/qwen2_vl_7b_pissa_qlora_128_fintabnet_en.yaml`

#### Supervised Fine-Tuning on Single Node

```
FORCE_TORCHRUN=1 llamafactory-cli train ./train_configs/train/qwen2_vl_7b_pissa_qlora_128_fintabnet_en.yaml
```

or use the Slurm sbatch. Example script here `./submit_train_singlenode.sh` for single node single GPU i.e. g5.2xlarge 

```
sbatch submit_train_singlenode.sh 
```

####  Supervised Fine-Tuning on Multiple Nodes
use the Slurm sbatch. Example script here `./submit_train_multinode.sh` for 2 nodes of single GPU i.e. 2 * g5.2xlarge 

```
sbatch submit_train_multinode.sh 
```
After completing a model training process, you'll get a file called `finetune_output_multinode.log`. This is a log file that records all the details and progress of your training session ([example here](https://github.com/aws-samples/fine-tune-qwen2-vl-with-llama-factory/blob/main/train_configs/finetune_output_multinode.log)).



### Export Models with merge LoRA 

Example `./train_configs/export/export_qwen2_vl_7b_pissa_qlora_128_fintabnet_en.yaml`

1. Modify the adapter_name_or_path  to your target lora folder path
2. Modify the output directory export_dir  to your target output folder path


```bash
llamafactory-cli export ./train_configs/export/export_qwen2_vl_7b_pissa_qlora_128_fintabnet_en.yaml
```


### [Optional] Quantization with AutoAWQ

AutoAWQ is an easy-to-use package for 4-bit quantized models. AutoAWQ speeds up models by 3x and reduces memory requirements by 3x compared to FP16. AutoAWQ implements the Activation-aware Weight Quantization (AWQ) algorithm for quantizing LLMs.

```bash
cd ..
git clone https://github.com/kq-chen/AutoAWQ.git
cd AutoAWQ
pip install numpy gekko pandas
pip install -e .
```

```bash
CUDA_VISIBLE_DEVICES=0 python ./quantization/quant_awq.py --model_path ./models/qwen2_vl_7b_pissa_qlora_128_fintabnet_en --quant_path ./models/qwen2_vl_7b_pissa_qlora_128_fintabnet_en_awq_int4 --jsonl_file ./data/fintabnet_en/fintabnet.json --n_sample 16
```

### Evaluation

#### Step 1. Inference

```bash
python ./evaluation/inference.py --log-path ./logs --model-name qwen2_vl --model-path models/qwen2_vl_7b_pissa_qlora_128_fintabnet_en
```

#### Step 2. Scoring

```bash
python ./evaluation/calc_teds.py ./logs/$YOUR_TXT_PATH
```

## Hosting

You can either host fine-tuned Qwen2 VL model on SageMaker real-time endpoint, or use directly vLLM docker on your perferred environment such as EKS. You can check the [deployment guidance here](https://github.com/aws-samples/fine-tune-qwen2-vl-with-llama-factory/tree/main/hosting_vllm)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

