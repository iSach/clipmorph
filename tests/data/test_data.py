import torch 

from clipmorph.data.__init__ import load_data, Data    
from torchvision import transforms as T
import os 

def test_data():
    """Test the data loading."""

    data_list = os.listdir('./training_data/styles')
    data_list.sort()

    data_path = './training_data/styles'
    data = Data(data_path)
    assert data.__len__() == len(data_list)
    assert data.num_images == len(data_list)
    assert data.root_dir  == data_path
    assert data.img_size ==  None
    prov = data.img_names
    prov.sort()
    assert prov == data_list

    for i in range(data.__len__()):
        img, img_name = data.__getitem__(i)
        assert img_name == data_list[i]  
       

    data = Data(data_path, 224)
    assert data.__len__() == len(data_list)
    assert data.num_images == len(data_list)
    assert data.root_dir  == data_path
    assert data.img_size ==  224
    prov = data.img_names
    prov.sort()
    assert prov == data_list

    data_loader = load_data(data_path, 7, 224)

    for batch in data_loader:
        assert batch.size(0) == 7
        assert type(batch) == torch.Tensor
        break  # Stop after one iteration


    