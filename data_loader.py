import torch.utils.data as data
from utils import line_to_tensor
import torch


class CityDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, cities, labels):
        """Set the path for audio data, together wth labels and objid.

        Args:

        """
        self.cities = cities
        self.labels = labels

    def __getitem__(self, index):
        """Returns one data pair (city and label)."""
        city = line_to_tensor(self.cities[index])
        label = self.labels[index]
        return city, label

    def __len__(self):
        return len(self.cities)
    
    
def collate_fn(data):
    # Sort a data list by length (descending order).
    data.sort(key=lambda x: x[0].shape[0], reverse=True)
    cities, labels = zip(*data)

    lengths = [city.shape[0] for city in cities]
    lengths = torch.LongTensor(sorted(lengths)[::-1])
    num_coeffs = data[0][0].shape[1]
    padded_cities = torch.zeros(len(cities), max(lengths), num_coeffs)
    for i, city in enumerate(cities):
        end = lengths[i]
        padded_cities[i, :end, :] = torch.from_numpy(city[:end])
        
    # Merge labels.
    labels = torch.FloatTensor(labels)
    return padded_cities, labels, lengths


def get_train_loader(cities, labels, batch_size, shuffle, sampler, collate_fn):
    """Returns torch.utils.data.DataLoader for custom dataset."""

    dataset = CityDataset(cities=cities, labels=labels)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn,
                                              sampler = sampler)
    return data_loader
