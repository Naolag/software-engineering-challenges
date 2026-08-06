def search_rotated_recursive(arr, target, left=0, right=None):
    if right is None:
        right = len(arr) - 1
    if left > right:
        return -1
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    
    # Left half sorted
    if arr[left] <= arr[mid]:
        if arr[left] <= target < arr[mid]:
            return search_rotated_recursive(arr, target, left, mid-1)
        else:
            return search_rotated_recursive(arr, target, mid+1, right)
    else:
        if arr[mid] < target <= arr[right]:
            return search_rotated_recursive(arr, target, mid+1, right)
        else:
            return search_rotated_recursive(arr, target, left, mid-1)
