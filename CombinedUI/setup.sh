#!/bin/bash

echo_error(){
    echo "Error occurred in step: $1"
    exit 1
}

repo_checkout(){
    #Repos to checkout
    repos=(
        "https://github.com/open-mmlab/mmcv.git"
        # "https://github.com/ViTAE-Transformer/ViTPose.git"
        "https://github.com/shubham-goel/4D-Humans.git FDHumans"
    )

    # Directory to clone repositories into
    clone_dir="./"

    # Create the directory if it doesn't exist
    mkdir -p "$clone_dir"

    # Navigate to the directory
    cd "$clone_dir" || { echo "Failed to enter directory $clone_dir"; exit 1; }

    # Function to clone a repository and optionally rename it
    clone_repo() {
        local repo_info="$1"
        local repo_url
        local custom_name

        # Split the repo_info into URL and custom name (if provided)
        repo_url=$(echo "$repo_info" | awk '{print $1}')
        custom_name=$(echo "$repo_info" | awk '{print $2}')

        # Extract the repo name from URL
        local repo_name
        repo_name=$(basename "$repo_url" .git)

        echo "Cloning $repo_name..."

        # Clone the repository
        if git clone "$repo_url"; then
            echo "Successfully cloned $repo_name."

            # Rename the repository directory if a custom name is provided
            if [[ -n "$custom_name" && "$custom_name" != "$repo_name" ]]; then
                mv "$repo_name" "$custom_name" && echo "Renamed $repo_name to $custom_name."
            fi
        else
            echo "Error cloning $repo_name. Skipping..."
        fi
    }

    # Loop through the repository list and clone each one
    for repo in "${repos[@]}"; do
        clone_repo "$repo"
    done

    echo "All repositories processed."
}

weights_checkout(){
    conda install curl -y
    rm -rf checkpoints
    mkdir -p checkpoints
    cd checkpoints
    curl -L -o vitpose_checkpoint.pth "https://62afda.dm.files.1drv.com/y4mBDiqHvl4ClkQbjljDfxZ35JemNwe-D-YlTuMfeya1BIR5tVP3cO26ntjrJkBL-2L8beSmOOPy7149gWRMkDqTZCPhS--XxryYZLSGtdKxR5ADq-9S_6ApoHxLbQP4MOs63iPz2jSLQMFqJFcFdoXZ2ml2HyvGkCu7MxyP9ELoZvtYRyipBDvsFvR2bN7xUknS6LR5HdBjGpZtM7saMmIXQ"
    cd ..
}

create_conda_env(){
    conda create --name posecompare python=3.10.13 -y
    eval "$(conda shell.bash hook)"
    conda activate posecompare
    conda install ipykernel -y
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
}

create_env_from_file(){
    conda env create -f environment.yml
    conda activate posecompare
}

mediapipe_setup(){
    pip install mediapipe
    pip install apex
}

vitpose_setup(){
    cd mmcv
    git checkout v1.3.9
    MMCV_WITH_OPS=1 pip install -e .
    cd ../ViTPose
    pip install -v -e .
    cd ..
    pip install timm==0.4.9 einops
}

verifiy_opengl(){
    if conda list | grep -q "osmesa"; then
        echo "osmesa is already installed in the current Conda environment."
    else
        echo "osmesa is not installed. Installing osmesa..."
        git clone https://github.com/mmatl/pyopengl.git
        pip install ./pyopengl
        # # Install osmesa in the current Conda environment
        # conda install -c conda-forge osmesa -y
        
        # if [ $? -eq 0 ]; then
        #     echo "osmesa successfully installed in the current Conda environment."
        # else
        #     echo "Failed to install osmesa. Please check your Conda environment and try again."
        #     exit 1
        # fi
    fi
}

fdhuman_setup(){
    cd FDHumans
    pip install -e .[all]
    cd ..
    verifiy_opengl || echo_error "opengl"
}


gradio_setup(){
    pip install -r requirements.txt
}

main(){
    echo "Starting setup process"
    repo_checkout || echo_error "repo_checkout"
    weights_checkout || echo_error "weights"
    create_conda_env || echo_error "create_conda_env"
    gradio_setup || echo_error "Gradio"
    mediapipe_setup || echo_error "mediapipe"
    # # vitpose_setup || echo_error "ViTPose"
    fdhuman_setup || echo_error "FDHumans"
    
    echo "Setup finished!"
}

main