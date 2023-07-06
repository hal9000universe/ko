#[cfg(test)]
use crate::cartesian_product;

#[test]
fn test_cartesian_product() {
    let a: Vec<i32> = vec![1, 2, 3];
    let b: Vec<i32> = vec![4, 5, 6];
    let test_product: Vec<Vec<i32>> = cartesian_product!(a, b);
    let correct_product: Vec<Vec<i32>> = vec![
        vec![1, 4],
        vec![1, 5],
        vec![1, 6],
        vec![2, 4],
        vec![2, 5],
        vec![2, 6],
        vec![3, 4],
        vec![3, 5],
        vec![3, 6],
    ];
    for test_elem in test_product.iter() {
        assert!(correct_product.contains(test_elem));
    }
    for correct_elem in correct_product.iter() {
        assert!(test_product.contains(correct_elem));
    }

    let a: Vec<i32> = vec![1, 2, 3];
    let b: Vec<i32> = vec![4, 5, 6];
    let vectors: &Vec<Vec<i32>> = &vec![a, b];
    let test_product: Vec<Vec<i32>> = cartesian_product!(vectors);
    let correct_product: Vec<Vec<i32>> = vec![
        vec![1, 4],
        vec![1, 5],
        vec![1, 6],
        vec![2, 4],
        vec![2, 5],
        vec![2, 6],
        vec![3, 4],
        vec![3, 5],
        vec![3, 6],
    ];
    for test_elem in test_product.iter() {
        assert!(correct_product.contains(test_elem));
    }
    for correct_elem in correct_product.iter() {
        assert!(test_product.contains(correct_elem));
    }
}
