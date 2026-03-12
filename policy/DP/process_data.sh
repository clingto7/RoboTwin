#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
episode_ids_file=${4}
subset_tag=${5}

if [ -n "$episode_ids_file" ]; then
    python process_data.py $task_name $task_config $expert_data_num --episode-ids-file "$episode_ids_file" --subset-tag "$subset_tag"
else
    python process_data.py $task_name $task_config $expert_data_num
fi
