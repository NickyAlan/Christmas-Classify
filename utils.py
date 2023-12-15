import os 
import torch
import random
import shutil
from PIL import Image
from timeit import default_timer as timer 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def to_jpg(image_path:str) :
    '''
    convert almost all image format(except .avif) to jpg format
    '''
    if image_path.endswith('avif') :
        print(f"can't convert the {image_path}\nbecause .avif format")
    if not image_path.endswith('jpg') and not image_path.endswith('avif') :
        image = Image.open(image_path).convert('RGB')
        image_name = image_path.split('.')[:-1]
        save_path = ''.join(image_name).strip('/')
        save_path = f'{save_path}.jpg'
        image.save(save_path)
        os.remove(image_path)

def create_dataset(raw_data_dir = 'raw_data', dataset_dir = 'dataset',
                  train_size = 0.8, val_size = 0.15, test_size = 0.05, hasTest = True, shuffle=True) : 

    class_names = os.listdir(raw_data_dir)
    for class_name in class_names :
        class_path = os.path.join(raw_data_dir, class_name)
        for image_name in os.listdir(class_path) :
            image_path = os.path.join(class_path, image_name)
            to_jpg(image_path)
        
    if hasTest :
        assert train_size + val_size + test_size <= 1  , 'all of folder must be 100%'
        assert train_size > val_size and train_size >= test_size , 'train folder size have to more than val and test folder'
        splitFolder = ['train', 'val', 'test']
    else :
        assert train_size + val_size  == 1 , 'all of folder must be 100%'
        assert train_size > val_size , 'train folder size have to more than val folder'
        splitFolder = ['train', 'val']

    if not os.path.isdir(dataset_dir) :
        print(f'creating {dataset_dir} folder ...')
        os.makedirs(dataset_dir)
        for folder in splitFolder :
            os.makedirs(os.path.join(dataset_dir, folder))

        for class_name in os.listdir(raw_data_dir) :
            main_folder = os.path.join(raw_data_dir, class_name)
            images_name = os.listdir(main_folder)
            total_image = len(images_name)
            
            if shuffle :
                random.shuffle(images_name)
            
            train_images = images_name[:int(train_size*total_image)]

            if hasTest :
                val_images = images_name[int(train_size*total_image):int(train_size*total_image)+int(val_size*total_image)]
                test_images = images_name[int(train_size*total_image)+int(val_size*total_image):]
            else :
                val_images = images_name[int(train_size*total_image):]
            
            for folder in splitFolder :
                os.makedirs(os.path.join(dataset_dir, folder, class_name))
            
            for image_name in train_images :
                source_path = os.path.join(raw_data_dir, class_name, image_name)
                desc_path = os.path.join(dataset_dir, 'train', class_name, image_name)
                shutil.copy(source_path, desc_path)
            
            for image_name in val_images :
                source_path = os.path.join(raw_data_dir, class_name, image_name)
                desc_path = os.path.join(dataset_dir, 'val', class_name, image_name)
                shutil.copy(source_path, desc_path)

            if hasTest :
                for image_name in test_images :
                    source_path = os.path.join(raw_data_dir, class_name, image_name)
                    desc_path = os.path.join(dataset_dir, 'test', class_name, image_name)
                    shutil.copy(source_path, desc_path)
        
    return class_names

def train_one_epoch(model, dataloader, loss_fn, optimizer) :
    model.train()
    train_acc, train_loss = 0, 0
    for batch, (images, labels) in enumerate(dataloader, start=1) :
         images, labels = images.to(device), labels.to(device)
         predict = model(images) 
         loss = loss_fn(predict, labels)
         train_loss += loss.item()

         optimizer.zero_grad()
         loss.backward()
         optimizer.step()

         predict_labels = torch.argmax(predict, dim=1) 
         train_acc += (predict_labels == labels).sum().item() / len(predict_labels)

         if batch ==1 or batch%5 == 0 or (batch == len(dataloader)) :
             print(f' {batch}/{len(dataloader)} batches')
    
    train_acc = train_acc / len(dataloader)
    train_loss = train_loss / len(dataloader) 

    return train_acc, train_loss

def test_one_epoch(model, dataloader, loss_fn) :
    model.eval()
    test_acc, test_loss = 0, 0
    with torch.inference_mode() :
        for batch, (images, labels) in enumerate(dataloader, start=1) :
            images, labels = images.to(device), labels.to(device)
            predict = model(images) 

            loss = loss_fn(predict, labels)
            test_loss += loss.item()

            predict_labels = torch.argmax(predict, dim=1)
            test_acc += (predict_labels == labels).sum().item() / len(predict_labels)

    test_acc = test_acc / len(dataloader)
    test_loss = test_loss / len(dataloader)

    return test_acc, test_loss

def train(
    model, train_dataloader, val_dataloader, loss_fn, optimizer, epochs, scheduler, model_name, class_names, patience
) :
    os.makedirs("save_weights", exist_ok=True)
    result = {
        'train_acc' : [],
        'train_loss' : [],
        'val_acc' : [],
        'val_loss' : [],
    }
    start_time = timer()
    # early stopping 
    best_valid_loss = float('inf')
    current_patience = 0
    problemClassifier = 'MultiClassifier' if len(class_names) > 2 else 'BinaryClassifier'
    print(f'\n\tstart training {model_name}({problemClassifier})')
    for epoch in range(1, epochs+1) :
        print(f'EPOCHS {epoch}/{epochs}')
        train_acc, train_loss = train_one_epoch(model, train_dataloader, loss_fn, optimizer)
        val_acc, val_loss = test_one_epoch(model, val_dataloader, loss_fn)
        scheduler.step(val_loss)
    
        if val_loss < best_valid_loss :
            best_valid_loss = val_loss 
            current_patience = 0 # reset patience early stopping
            save_path = f'save_weights/best_{model_name}_weights.pt'
            torch.save(model.state_dict(), save_path)
            print(f'\n\tsave {model_name} with validation loss = {val_loss:.6f}')
            
            best_epoch = f'best epoch {epoch} val_acc {val_acc*100:.2f}%  val_loss {val_loss:.4f}'
        
        else : 
            current_patience += 1

        print(f'''
        \ttrain_acc {train_acc*100:.2f}% | val_acc {val_acc*100:.2f}%
        \ttrain_loss {train_loss:.4f} | val_loss {val_loss:.4f}
        \tEarly stopping: {current_patience}/{patience}
        ''')

        # Check early stopping condition
        if current_patience >= patience:
            print(f'Early stopping after {epoch} epochs.')
            break

        result['train_acc'].append(train_acc)
        result['train_loss'].append(train_loss)
        result['val_acc'].append(val_acc)
        result['val_loss'].append(val_loss)
    
    end_time = timer()
    total_time = (end_time-start_time) / 60
    print(f"Total training time: {total_time:.3f} minutes")
    print(f'[INFO] {best_epoch}')

    return result, total_time