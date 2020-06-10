from torchvision import transforms

transform1 = transforms.Compose([
    # transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225]),
    # transforms.Normalize(mean = [0.4942, 0.4686, 0.4322], std = [0.1123, 0.1160, 0.1416])
    transforms.Normalize(mean = [0., 0., 0.], std = [1, 1, 1])
    # transforms.Normalize(mean = [0.4899, 0.4894, 0.4590], std = [0.1114, 0.1159, 0.1385])    
    # transforms.Normalize(mean = [0.4737, 0.4622, 0.4366], std = [0.0933, 0.1039, 0.1308])
    ])

cut = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    ])

convert = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.Normalize(mean = [0.4942, 0.4686, 0.4322], std = [0.1123, 0.1160, 0.1416])
    transforms.Normalize(mean = [0., 0., 0.], std = [1, 1, 1])
    # transforms.Normalize(mean = [0.4899, 0.4894, 0.4590], std = [0.1114, 0.1159, 0.1385]) 
    # transforms.Normalize(mean = [0.4737, 0.4622, 0.4366], std = [0.0933, 0.1039, 0.1308])
    ])

transform = transforms.Compose([
    # cut.transforms[0],
    # cut.transforms[1],
    convert.transforms[0],
    convert.transforms[1]
    ])
