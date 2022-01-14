import chemprop

if __name__ == '__main__':
    freeze_support()
    
arguments = [
    '--data_path', '/Users/agubo/Desktop/Exercise2/trainset.csv',
    '--dataset_type', 'regression',
    '--save_dir', 'Ex3_checkpoints'
]

args = chemprop.args.TrainArgs().parse_args(arguments)
mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)