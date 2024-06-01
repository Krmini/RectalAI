def pip_install_all():
    import subprocess
    import time
    pip_list = [['pip', 'install', 'git+https://github.com/tensorflow/docs'],
                ['pip', 'install', 'imageio'],
                ['pip', 'install', 'tqdm'],
                ['apt-get', 'install', '-y', 'libopenexr-dev'],
                ['pip', 'install', '--upgrade OpenEXR'],
                ['pip', 'install', 'tensorflow_graphics'],
                ['pip', 'install', 'tensorflow-addons'],
                ['pip', 'install', 'torch'],
                ['pip', 'install', 'torchvision'],
                ['pip', 'install', 'grad-cam'],
                ['pip', 'install', 'PyYAML'],
                ['pip', 'install', 'piq'],
                ['pip', 'install', 'torchcam'],
                ['pip', 'install', 'scikit-learn'],
                ['apt-get', 'update'],
                ['apt install', '-y', 'ffmpeg', 'libsm6', 'libxext6']
                ]
    print('Cammands to be run:')
    [print(i) for i in pip_list]  # Print the list before asking for input

    time.sleep(1)
    
    run_pip_install = input('Do you want to run pip install? [y]')
    if run_pip_install.lower() == 'y':
        for command in pip_list:
            subprocess.run(command)


def load_dataset(positive = True, custom_path = None):
    """
    Indlæser dataset med hhv positiv eller negativ cases afhængigt af "positive".
    De do datasæt kan samles med ds1.concatenate(ds2)
    """
    from tensorflow.data import Dataset

    if custom_path != None:
        path = custom_path
    elif positive == True:
        path = "/tf/data/cropped/1"
    else:
        path = "/tf/data/cropped/0"
        
    
    tfrecord_filepath = path
    with tf.device('/CPU:0'):
        ds = Dataset.load(tfrecord_filepath)
    print('Loaded from:', path)
    return ds
