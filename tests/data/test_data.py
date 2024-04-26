import torch 

from clipmorph.data.__init__ import load_data, Data    
from PIL import Image
from torchvision import transforms as T


def test_data():
    """Test the data loading."""

    data_list = ['klee.jpg', 'starrynight.jpg', 'lines.jpg', 'scream.jpg', 'basquiat.jpg', 'simpsons.jpg', 'dali.jpg', 'monet.jpg', 'ermine.jpg', 'chagall.jpg', 'kandinsky.jpg', 'okeeffe.jpg', 'watercolor.jpg', 'femme.jpg', 'hopper.jpg', 'klimt.jpg', 'kanagawa.jpg', 'muse.jpg', 'picasso.jpg', 'sketch.jpg', 'turner.jpg']

    data_path = './training_data/styles'
    data = Data(data_path)
    assert data.__len__() == 21
    assert data.num_images == 21
    assert data.root_dir  == data_path
    assert data.img_size ==  None
    assert data.img_names.sort() == data_list.sort()

    for i in range(data.__len__()):
        img, img_name = data.__getitem__(i)
        assert img_name == data_list[i]  
       

    data = Data(data_path, 224)
    assert data.__len__() == 21
    assert data.num_images == 21
    assert data.root_dir  == data_path
    assert data.img_size ==  224
    assert data.img_names == data_list

    data_loader = load_data(data_path, 7, 224)

    for batch in data_loader:
        assert batch.size(0) == 7
        assert type(batch) == torch.Tensor
        break  # Stop after one iteration


    """# Test two images
    def test_im(img, name, size):
        im_o = Image.open('./training_data/styles/'+name, mode='r').convert('RGB')
        if size is not None:
            transfo = T.Compose([
                    T.Resize(size),
                    T.CenterCrop(size),
                    T.ToTensor(),
                    T.Lambda(lambda x: x.mul(255))
                ])
        else:
            transfo = T.Compose([
                T.ToTensor(),
                T.Lambda(lambda x: x.mul(255))
            ])

        im_o = transfo(im_o)
        for j in range(im_o.size()[0]):
            for k in range(im_o.size()[1]):
                for l in range(im_o.size()[2]):
                    assert img[j][k][l] == im_o[j][k][l]  """