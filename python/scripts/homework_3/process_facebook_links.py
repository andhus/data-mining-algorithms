from __future__ import print_function, division


from data_mining.graph import TriestBase

from data_mining.data import get_facebook_links

if __name__ == '__main__':
    seed = None
    fbl = get_facebook_links(unique=True)

    tb_reference = TriestBase(size=len(fbl))
    tb_reference(fbl)

    tb = TriestBase(size=10000)
    tb(fbl)

    true_number_triangles = tb_reference.get_estimated_num_triangles()
    est_number_triangles = tb.get_estimated_num_triangles()

    print("true num triangles: {}".format(true_number_triangles))
    print("estimated num triangles: {}".format(est_number_triangles))
