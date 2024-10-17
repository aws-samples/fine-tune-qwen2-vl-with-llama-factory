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


### Miniconda install 
```
get https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -f -p ~/miniconda3
source ~/miniconda3/bin/activate
```

### Create environment on worker node (g5.2xlarge)
```
conda create -n llamafactory python=3.10 
conda activate llamafactory
```
```
pip install -e ".[torch,metrics,deepspeed,bitsandbytes,liger-kernel]" "transformers>=0.45.0"
pip install flash-attn
```

### Start the data pre-processing
copy the  `process_data.py` file into LLaMA-Factory and execute

```
cd LLaMA-Factory
python process_data.py --output_dir ./data/pubtabnet
```

Add pubtabnet format in `./data/dataset_info.json`

### Start the training job

Prepare training config `./train_configs/qwen2_vl_7b_sft_cfg.yaml`

#### Supervised Fine-Tuning on Single Node

```
FORCE_TORCHRUN=1 llamafactory-cli train ./train_configs/qwen2_vl_7b_sft_cfg.yaml
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


###  Merge LoRA

Example `./lora_merge_configs/qwen2vl_lora_sft.yaml`

1. Modify the adapter_name_or_path  to your target lora folder path
2. Modify the output directory export_dir  to your target output folder path

```
llamafactory-cli export ./lora_merge_configs/qwen2vl_lora_sft.yaml  
```

###  Evaluation using Hugging Face library 

You can process the testing data and run the evaluation 
```
cd qwen2vl_evaluation
python qwen2vl_evaluation.py
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

