
#fitness functions
SSE=0
INTRA_CLUSTER_DISTANCE=1
QUANTIZATION_ERROR=2
SI=3
DBI=4


#Initialization Help
RANDOM_NUMBER=-1
RANDOM_INITIALIZATION=0
USE_KMEANS=1
KMEANS_PLUSE_PLUSE=2
HYPRID=4


def get_initialization_name(initializaion_type):
    if initializaion_type==RANDOM_NUMBER:
        return "RANDOM NUMBER In the space"
    elif initializaion_type==RANDOM_INITIALIZATION:
        return "RANDOM DATA POINT "
    elif initializaion_type == USE_KMEANS:
        return "USE K-MEANS"
    elif initializaion_type == KMEANS_PLUSE_PLUSE:
        return "K-MEANS++ "
    else:
        return "HYBRID "

def get_fitness_function_name(fitness_function):
    if fitness_function == SSE:
        return "SUM of Square Errors"
    elif fitness_function == INTRA_CLUSTER_DISTANCE:
        return "INTRA-CLUSTER DISTANCE "
    elif fitness_function == QUANTIZATION_ERROR:
        return "QUANTIZATION ERROR"
    elif fitness_function == SI:
        return "Silhouette Index"
    elif fitness_function == DBI:
        return "DAVIES BOULDIN SCORE"
    else:
        return "UNKNOWN "

