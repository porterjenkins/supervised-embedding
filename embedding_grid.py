from unsupervised_embedding import *

def grid_search(min,max):

    for i in range(min, max+1):
        batch_size = 32
        embedding_size = 50
        skip_window = i
        num_skips = 2
        num_sampled = 16
        num_steps = 5000
        dao = DatabaseAccess(city='jinan',data_dir="/Volumes/Porter's Data/penn-state/data-sets/")
        main(dao, batch_size, embedding_size, skip_window, num_skips, num_sampled, num_steps)

if __name__ == '__main__':
    grid_search(min=2,max=6)