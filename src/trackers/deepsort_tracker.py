from deep_sort_realtime.deepsort_tracker import DeepSort

def create_tracker(max_age=60, n_init=2, nms_max_overlap=0.7, embedder=None):
    return DeepSort(max_age=max_age, n_init=n_init, nms_max_overlap=nms_max_overlap, embedder=embedder)
