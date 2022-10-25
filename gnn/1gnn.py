import torch
import numpy
from scipy import sparse
from torch_sparse import SparseTensor

'''in numpy with SPARSE'''

matrix=numpy.array([[0,0,1,1],[1,0,1,0],[0,1,0,0]])
print(matrix)

sparse_matrix=sparse.coo_matrix(matrix)
print(sparse_matrix)

matrix_row_col=[sparse_matrix.row,sparse_matrix.col]
print(matrix_row_col)

print('=========🔦=========')

'''using pytorch'''

tensor=torch.LongTensor(numpy.array(matrix_row_col))
print(tensor)

sparse_tensor=SparseTensor(row=tensor[0],col=tensor[1],sparse_sizes=(3,4))
print(sparse_tensor)

print(sparse_tensor.to_dense())

print('=========================')

def convert_r_mat_edge_index_to_adj_mat_edge_index(input_edge_index, row_size, col_size):
    R = torch.zeros((row_size, col_size))
    
    # convert sparse coo format to dense format to get R
    for i in range(len(input_edge_index[0])):
        row_idx = input_edge_index[0][i]
        col_idx = input_edge_index[1][i]
        R[row_idx][col_idx] = 1

    # perform the  r_mat to adj_mat conversion   
    R_transpose = torch.transpose(R, 0, 1)
    adj_mat = torch.zeros((row_size + col_size , row_size + col_size))
    adj_mat[: row_size, row_size :] = R.clone()
    adj_mat[row_size :, : row_size] = R_transpose.clone()
    
    # convert from dense format back to sparse coo format so we can get the edge_index of adj_mat
    adj_mat_coo = adj_mat.to_sparse_coo()
    adj_mat_coo = adj_mat_coo.indices()
    return adj_mat_coo  

print(100)
adj_mat_edge_index = convert_r_mat_edge_index_to_adj_mat_edge_index(tensor, row_size=3, col_size=4)
print(adj_mat_edge_index)    

adj_mat = SparseTensor(row=adj_mat_edge_index[0], 
                                       col=adj_mat_edge_index[1], 
                                       sparse_sizes=(3+4, 4+3))

print(adj_mat.to_dense())


def convert_adj_mat_edge_index_to_r_mat_edge_index(input_edge_index, row_size, col_size):
    # create a sparse tensor so we can easily do the to_dense conversion and get a sub matrix to 
    # get R (interaction matrix) and then convert it back to sparse coo format
    sparse_input_edge_index = SparseTensor(row=input_edge_index[0], 
                                           col=input_edge_index[1], 
                                           sparse_sizes=((row_size + col_size), row_size + col_size))
    adj_mat = sparse_input_edge_index.to_dense()
    interact_mat = adj_mat[: row_size, row_size :]
    r_mat_edge_index = interact_mat.to_sparse_coo().indices()
    return r_mat_edge_index



converted_back_to_r_mat_edge_index = convert_adj_mat_edge_index_to_r_mat_edge_index(adj_mat_edge_index, 3, 4)

converted_back_to_r_mat = SparseTensor(row=converted_back_to_r_mat_edge_index[0], 
                                       col=converted_back_to_r_mat_edge_index[1], 
                                       sparse_sizes=(3, 4))

print(converted_back_to_r_mat.to_dense())