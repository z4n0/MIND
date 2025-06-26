from monai.transforms import MapTransform

class PrintShapeTransform(MapTransform): #inherit from MapTransform
    """
    A simple transform that prints the shape and min/max of 'image'
    (or any key passed when it's created) to help debug the data pipeline.
    """
    #constructor
    def __init__(self, keys, tag="", print_stats=False, allow_missing_keys=False):
        """
        Args:
            keys: which keys to process. e.g. ["image"] or ["image", "label"].
            tag: optional string to identify this stage in logs (ie it tells which tranform has just been performed).
            allow_missing_keys: if False, will raise error if key not found.
        example usage:
            transform = PrintShapeTransform(keys=["image"], tag="before")
        """
        super().__init__(keys, allow_missing_keys)
        self.tag = tag #private variable
        self.print_stats = print_stats #private variable 
    # tells that this class is callable
    # take the dictionary of data, for each key specified in the constructor
    # gets the tensor data, convert it to numpy array, print shape and range
    def __call__(self, data):
        d = dict(data) #convert data to dictionary
        for key in self.key_iterator(d):
            # d[key] should be a torch.Tensor or MetaTensor
            # shape is (C, H, W) typically
            tensor = d[key]
            array = tensor.cpu().numpy()  # move to CPU if needed
            shape = array.shape
            min_val, max_val = float(array.min()), float(array.max())
            mean_val = float(array.mean())
            if self.print_stats:
                print(
                    f"[{self.tag}] \nshape={shape}\n"
                    f"range=[{min_val:.3f},{max_val:.3f}]\n"
                    f"mean=[{mean_val:.3f}]\n"
                )
        return d