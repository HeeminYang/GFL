from utils import fed_avg

import os
import time
import copy
from tqdm import tqdm

# client run GAN
def run(args, client_list, save_path, W, logger):
    init_W = copy.deepcopy(W)
    one_round_g_epoch = int(args.g_epoch / args.warm_up)

    # Round start
    for Round in range(args.round):
        logger.info(f'Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
        logger.info(f'Round {Round+1} start')
                
        # GAN training
        logger.info('GAN training')
        for i, client in enumerate(client_list):
            logger.info(f'Client {i+1}')
            if not os.path.exists(f'{save_path}/{args.project_name}/img/client{i+1}'):
                os.makedirs(f'{save_path}/{args.project_name}/img/client{i+1}')
            client.gan_train(f'{save_path}/{args.project_name}/img/client{i+1}', one_round_g_epoch, Round)
            if not os.path.exists(f'{save_path}/{args.project_name}/model/client{i+1}'):
                os.makedirs(f'{save_path}/{args.project_name}/model/client{i+1}')
            client.save_model(f'{save_path}/{args.project_name}/model/client{i+1}')

        # Make generated data loaders

        # local training
        logger.info('Local Classifier training')
        for i, client in enumerate(client_list):
            logger.info(f'Client {i+1}')
            client.local_train()

        # local test
        if (Round + 1) % args.warm_up == 0:
            logger.info('Generate images')
            generated_loaders = []
            for i in tqdm(range(len(client_list)), desc='Generate images'):
                generated_loaders.append(copy.deepcopy(client_list[i].generate_image(f'{save_path}/{args.project_name}/img/client{i+1}', Round)))

            logger.info('Local Classifier test')
            W = copy.deepcopy(init_W)
            for i, client in enumerate(client_list):
                if not os.path.exists(f'{save_path}/{args.project_name}/wrong/client{i+1}'):
                    os.makedirs(f'{save_path}/{args.project_name}/wrong/client{i+1}')
                client.local_test(generated_loaders, f'{save_path}/{args.project_name}/wrong/client{i+1}', Round, W[i], i)
        
            logger.info(f'Changed Adjacency Matrix: \n{W}')

        # classifier training
        logger.info('Classifier training')
        for i, client in enumerate(client_list):
            logger.info(f'Client {i+1}')
            client.clf_train()
        
        # Aggregation
        logger.info('Aggregation')
        for i, client in enumerate(client_list):
            # if not i > (args.client - args.malicious - 1):
            logger.info(f'Client {i+1}')
            client.modelcopy(copy.deepcopy(fed_avg(client_list, W[i])))
            client.aggregation_test()
            # else:
            #     logger.info(f'Malicious Client {i+1} do train')
            #     client.clf_train()

# client run GAN
def simpleFL(args, client_list, save_path, W, logger):
    # Round start
    for Round in range(args.round):
        logger.info(f'Round {Round+1} start')
                
        # classifier training
        logger.info('Classifier training')
        for i, client in enumerate(client_list):
            logger.info(f'Client {i+1}')
            client.clf_train()
        
        # Aggregation
        logger.info('Calculate Aggregation model')
        for i, client in tqdm(enumerate(client_list)):
            # if not i > (args.client - args.malicious - 1):
            client.modelcopy(copy.deepcopy(fed_avg(client_list, W[i])))
            # else:
            #     logger.info(f'Malicious Client {i+1} do train')
            #     client.clf_train()
            #     client.aggregation_test()
        
        logger.info('Apply Aggregation model')
        for i, client in enumerate(client_list):
            logger.info(f'Client {i+1}')
            client.modify_clf()
            client.aggregation_test()

        # Result Graph
        logger.info('Result Graphing')
        for i, client in tqdm(enumerate(client_list)):
            if not os.path.exists(f'{save_path}/{args.project_name}/result/client{i+1}'):
                os.makedirs(f'{save_path}/{args.project_name}/result/client{i+1}')
            client.make_plot(f'{save_path}/{args.project_name}/result/client{i+1}')

# client run GAN
def proto(args, client_list, save_path, logger):
    # # K-nn training
    # logger.info('K-nn training')
    # for client in tqdm(client_list, desc='K-nn training'):
    #     client.knn_train()

    # check trained model is saved
    logger.info('Check trained model')
    if not os.path.exists(f'{save_path}/{args.project_name}/model/'):
        logger.info('Trained model is not saved')
        # GAN training
        logger.info('GAN training')
        for i, client in enumerate(client_list):
            logger.info(f'Client {i+1}')
            if not os.path.exists(f'{save_path}/{args.project_name}/img/client{i+1}'):
                os.makedirs(f'{save_path}/{args.project_name}/img/client{i+1}')
            client.gan_train(f'{save_path}/{args.project_name}/img/client{i+1}')
        
        # Save GAN model
        logger.info('Save GAN model')
        for i, client in tqdm(enumerate(client_list), desc='Save GAN model'):
            if not os.path.exists(f'{save_path}/{args.project_name}/model/client{i+1}'):
                os.makedirs(f'{save_path}/{args.project_name}/model/client{i+1}')
            client.save_model(f'{save_path}/{args.project_name}/model/client{i+1}')
    else:
        # Load GAN model
        logger.info('Load GAN model')
        for i, client in tqdm(enumerate(client_list), desc='Load GAN model'):
            client.load_model(f'{save_path}/{args.project_name}/model/client{i+1}')
    
    # Make generated data loaders
    logger.info('Generate images')
    generated_loaders = []
    for i in tqdm(range(len(client_list)), desc='Generate images'):
        generated_loaders.append(copy.deepcopy(client_list[i].generate_image(f'{save_path}/{args.project_name}/img/client{i+1}')))

    # classifier training
    logger.info('Classifier training')
    for i, client in enumerate(client_list):
        logger.info(f'Client {i+1}')
        if not os.path.exists(f'{save_path}/{args.project_name}/wrong/client{i+1}'):
            os.makedirs(f'{save_path}/{args.project_name}/wrong/client{i+1}')
        client.clf_train(copy.deepcopy(generated_loaders), f'{save_path}/{args.project_name}/wrong/client{i+1}')

    # # K-nn test
    # logger.info('K-nn test')
    # for i, client in enumerate(client_list):
    #     logger.info(f'Client {i+1}')
    #     if not os.path.exists(f'{save_path}/{args.project_name}/knn/client{i+1}'):
    #         os.makedirs(f'{save_path}/{args.project_name}/knn/client{i+1}')
    #     client.knn_test(copy.deepcopy(generated_loaders), f'{save_path}/{args.project_name}/knn/client{i+1}', i+1)