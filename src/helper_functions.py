def get_single_sample_by_index(data, i):
    all_data = data.__iter__().get_next()
    samples = all_data[0].numpy()
    labels = all_data[1].numpy()

    label_names = ['World', 'Sports', 'Business', 'Schience and Technology']
    return samples[i], labels[i], label_names[labels[i]]
