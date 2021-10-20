import os
import time
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import Image


dtype = torch.FloatTensor



def compute_mae_and_mse(model, data_loader, device):
    CS, mae, mse, num_examples = 0., 0., 0., 0
    for i, (features, targets) in enumerate(data_loader):

        features = features.to(device).type(dtype)
        targets = targets.to(device).type(dtype)
        targets = targets.unsqueeze(1)
        

        predictions = model(features)
        num_examples += targets.size(0)
        abs_error = torch.abs(predictions - targets)
        #counting the number of predictions which is off by 5
        CS += torch.sum(abs_error <=  5)
        mae += torch.sum(abs_error)
        mse += torch.sum((predictions - targets)**2)

    CS = CS.float()/num_examples
    mae = mae.float()/num_examples
    mse = mse.float()/num_examples


    return CS, mae, mse


def train(model, num_epochs, train_loader, validation_loader,
        test_loader, loss_fn, optimizer, device, len_train_dataset,
        batch_size, LOGFILE,TEST_VALID_LOSS, PATH, scheduler=None):
    
    start_time = time.time()
    
    best_CS = 9999
    best_mae = 9999
    best_rmse = 9999
    best_epoch	 = -1

    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            
            #get the inputs
            features = features.to(device).type(dtype)
            targets = targets.to(device).type(dtype)
            targets = targets.unsqueeze(1)

            # FORWARD AND BACK PROP
            predictions = model(features)
            cost = loss_fn(predictions, targets)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # backward pass
            cost.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
        #    print(f'batch:{batch_idx}')
            if not batch_idx % 25:
                s = ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                    % (epoch+1, num_epochs, batch_idx,
                        len_train_dataset//batch_size, cost))
                print(s)
                with open(LOGFILE, 'a') as f:
                    f.write('%s\n' % s)
        #scheduler.step()
        model.eval()
        with torch.set_grad_enabled(False):

            train_CS, train_mae, train_mse = compute_mae_and_mse(model,
                    train_loader, device)
            valid_CS, valid_mae, valid_mse = compute_mae_and_mse(model,
                    validation_loader, device)

        s = ('%d, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (epoch, train_CS, train_mae,\
                torch.sqrt(train_mse), valid_CS, valid_mae,\
                torch.sqrt(valid_mse)))

        print(f'Training loss: {train_mae} | Validation loss: {valid_mae}')
        with open(TEST_VALID_LOSS, 'a') as f:
            f.write('%s\n' % s)

        if valid_mae < best_mae:
            best_CS = valid_CS
            best_mae = valid_mae
            best_rmse = torch.sqrt(valid_mse)
            best_epoch = epoch
            ########## SAVE MODEL #############
            torch.save(model.state_dict(), os.path.join(PATH, 'best_model.pt'))


        s = 'CS/MAE/RMSE: | Current Valid: %.2f/%.2f/%.2f Ep. %d | Best Valid : %.2f/%.2f/%.2f Ep. %d' % (valid_CS, valid_mae,\
                torch.sqrt(valid_mse),epoch, best_CS, best_mae, best_rmse,\
                best_epoch)
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)
        

        s = 'Time elapsed: %.2f min' % ((time.time() - start_time)/60)
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)

    model.eval()
    with torch.set_grad_enabled(False):  # save memory during inference

        train_CS, train_mae, train_mse = compute_mae_and_mse(model, train_loader,
                                                device)
        valid_CS, valid_mae, valid_mse = compute_mae_and_mse(model,
                                         validation_loader, device)
        test_CS, test_mae, test_mse = compute_mae_and_mse(model, test_loader,
                                                device)

        s = 'CS/MAE/RMSE: | Train: %.2f/%.2f/%.2f | Valid: %.2f/%.2f/%.2f |\
                Test: %.2f/%.2f/%.2f' % (train_CS, train_mae, torch.sqrt(train_mse),
                       valid_CS, valid_mae, torch.sqrt(valid_mse),
                       test_CS, test_mae, torch.sqrt(test_mse))
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)

    s = 'Total Training Time: %.2f min' % ((time.time() - start_time)/60)
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)



def evaluation_of_best_model(model, train_loader, valid_loader, test_loader, LOGFILE,
        PATH,  device):

    ########## EVALUATE THE BEST MODEL###################

    model.load_state_dict(torch.load(os.path.join(PATH, 'best_model.pt')))
    model.eval()

    with torch.set_grad_enabled(False):

        train_CS, train_mae, train_mse = compute_mae_and_mse(model, train_loader,
                                                device)
        valid_CS, valid_mae, valid_mse = compute_mae_and_mse(model,
                                            valid_loader, device)
        test_CS, test_mae, test_mse = compute_mae_and_mse(model, test_loader,
                                                device)
        s = 'CS/MAE/RMSE: | Best Train: %.2f/%.2f/%.2f | Best Valid: %.2f/%.2f/%.2f |\
                Best Test: %.2f/%.2f/%.2f' % (train_CS, train_mae, torch.sqrt(train_mse),
                        valid_CS, valid_mae, torch.sqrt(valid_mse),
                        test_CS, test_mae, torch.sqrt(test_mse))
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)



########## SAVE PREDICTIONS ######

def save_predictions(model, test_loader,TEST_PREDICTIONS,  device):

    all_pred = []
    with torch.set_grad_enabled(False):
        for batch_idx, (features, targets) in enumerate(test_loader):

            features = features.to(device)
            predictions = model(features)
            lst = [str(int(i)) for i in  predictions]
            all_pred.extend(lst)

    with open(TEST_PREDICTIONS, 'w') as f:
        all_pred = ','.join(all_pred)
        f.write(all_pred)

