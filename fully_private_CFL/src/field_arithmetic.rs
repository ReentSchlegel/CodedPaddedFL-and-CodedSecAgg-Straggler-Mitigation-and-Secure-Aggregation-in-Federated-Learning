
use crate::MODULOS;
use crate::Field;
use crate::NUM_FRAC_BITS;
use rayon::prelude::*;



#[allow(unused)]
pub fn from_num(a: f64) -> Field {
    let b = a *2.0_f64.powf(NUM_FRAC_BITS as f64);
    let mut b = b.round() as i64;
    while b < 0 {
        b = b + (MODULOS as i64);
    }
    while b >= MODULOS as i64 {
        b = b - MODULOS as i64;
    }
    b as Field
}

#[cfg(test)]
mod from_num_tests {
    use super::*;
    #[test]
    fn from_num1() {
        assert_eq!(MODULOS-2_u64.pow(NUM_FRAC_BITS as u32), from_num(-1.0));
    }
    #[test]
    fn from_num2() {
        assert_eq!(2*2_u64.pow(NUM_FRAC_BITS as u32), from_num(2.0));
    }
    #[test]
    fn from_num3() {
        assert_eq!(0, from_num(0.0));
    }
}

#[allow(unused)]
pub fn to_float(a: Field) -> f64 {
    let range = 2_u64.pow((crate::NUM_TOTAL_BITS - 1) as u32);
    if a >= range && a < MODULOS - range {
        println!("Converting nonsense to float! {} {} {}",range,a,MODULOS-range);
    }
    let b:i64;
    if a >= (MODULOS-1)/2 {
        b = (a as i64) - (MODULOS as i64);
    } else {
        b = a as i64;
    }
    (b as f64) * 2.0_f64.powf(-1.0*(NUM_FRAC_BITS as f64))
}

#[cfg(test)]
mod to_float_tests {
    use super::*;
    #[test]
    fn to_float1() {
        assert_eq!(-1.0, to_float(from_num(-1.0)));
    }
    #[test]
    fn to_float2() {
        assert_eq!(4.0, to_float(from_num(4.0)));
    }
    #[test]
    fn to_float3() {
        assert_eq!(2.5, to_float(from_num(2.5)));
    }
}
#[allow(unused)]
pub fn multiply(a: Field, b: Field) -> Field {
    
    /*
    let two_p_f_inv: u64 = modinverse(2_i64.pow(NUM_FRAC_BITS as u32),MODULOS as i64).unwrap() as u64;
    (((a * b) % MODULOS) * two_p_f_inv) % MODULOS
    */

    //from_num(to_float(a)*to_float(b))

    let signed_a: i128;
    if a >= (MODULOS-1)/2 {
        signed_a = a as i128 - MODULOS as i128;
    } else {
        signed_a = a as i128;
    }

    let signed_b: i128;
    if b >= (MODULOS-1)/2 {
        signed_b = b as i128 - MODULOS as i128;
    } else {
        signed_b = b as i128;
    }

    /*
    let signed_prod = (signed_a * signed_b) as f64 * 2.0_f64.powf(-1.0* NUM_FRAC_BITS as f64);
    let mut signed_prod = signed_prod.round() as i64;

    while signed_prod < 0 {
        signed_prod += MODULOS as i64;
    }
    signed_prod as Field % MODULOS
    */

    
    let mut signed_prod = (signed_a * signed_b) / 2_i128.pow(NUM_FRAC_BITS as u32);

    while signed_prod < 0 {
        signed_prod += MODULOS as i128;
    }
    (signed_prod % MODULOS as i128) as Field
    
}

#[cfg(test)]
mod multiply_tests {
    use super::*;
    #[test]
    fn multiply1() {
        assert_eq!(-8.0, to_float(multiply(from_num(-2.0),from_num(4.0))));
    }
    #[test]
    fn multiply2() {
        assert_eq!(0.0, to_float(multiply(from_num(0.0),from_num(2.0))));
    }
    #[test]
    fn multiply3() {
        assert_eq!(4.0, to_float(multiply(from_num(1.0),from_num(4.0))));
    }
    #[test]
    fn multiply4() {
        assert_eq!(9.0, to_float(multiply(from_num(-3.0),from_num(-3.0))));
    }
}
#[allow(unused)]
pub fn divide(a: Field, b: Field) -> Field {
    
    /*
    let b_inv = modinverse(b as i64,MODULOS as i64).unwrap() as u64;
    let c = (a * b_inv) % MODULOS;
    (c * 2_u64.pow(NUM_FRAC_BITS as u32)) % MODULOS
    */

    /*
    let b_inv = from_num(1.0/to_float(b));
    multiply(a,b_inv)
    */

    //from_num(to_float(a)/to_float(b))

    let signed_a: i64;
    if a >= (MODULOS-1)/2 {
        signed_a = a as i64 - MODULOS as i64;
    } else {
        signed_a = a as i64;
    }

    let signed_b: i64;
    if b >= (MODULOS-1)/2 {
        signed_b = b as i64 - MODULOS as i64;
    } else {
        signed_b = b as i64;
    }

    let signed_quot = (signed_a as f64 / signed_b as f64) * 2.0_f64.powf(NUM_FRAC_BITS as f64);

    let mut signed_quot = signed_quot.round() as i64;

    while signed_quot < 0 {
        signed_quot += MODULOS as i64;
    }
    signed_quot as Field % MODULOS
}

#[cfg(test)]
mod divide_tests {
    use super::*;
    #[test]
    fn divide1() {
        assert_eq!(0.5, to_float(divide(from_num(1.0),from_num(2.0))));
    }
    #[test]
    fn divide2() {
        assert_eq!(2.0, to_float(divide(from_num(2.0),from_num(1.0))));
    }
    #[test]
    fn divide3() {
        assert_eq!(-2.0, to_float(divide(from_num(-4.0),from_num(2.0))));
    }
}

pub fn subtract(a: Field, b: Field) -> Field {
    let mut c:i64 = ((a as i64) - (b as i64)) % (MODULOS as i64);
    if c < 0 {
        c = c + MODULOS as i64;
    }
    c as u64
}

#[cfg(test)]
mod subtract_tests {
    use super::*;
    #[test]
    fn subtract1() {
        assert_eq!(-4.0, to_float(subtract(from_num(-1.0),from_num(3.0))));
    }
    #[test]
    fn subtract2() {
        assert_eq!(2.0, to_float(subtract(from_num(2.0),from_num(0.0))));
    }
    #[test]
    fn subtract3() {
        assert_eq!(5.0, to_float(subtract(from_num(3.0),from_num(-2.0))));
    }
    #[test]
    fn subtract4() {
        assert_eq!(-3.0, to_float(subtract(from_num(1.0),from_num(4.0))));
    }
}

#[allow(unused)]
pub fn vec_vec_add(a: &Vec<Field>, b: &Vec<Field>) -> Vec<Field>{
    a.iter().zip(b.iter()).map(|(ai,bi)| (*ai + *bi) % MODULOS).collect()
}

#[cfg(test)]
mod vec_vec_add_tests {
    use super::*;
    #[test]
    fn vec_vec_add1() {
        let a = vec![from_num(1.0),from_num(2.0)];
        let b = vec![from_num(1.0),from_num(1.0)];
        let c = vec![from_num(2.0),from_num(3.0)];
        assert_eq!(c,vec_vec_add(&a,&b));
    }
}
#[allow(unused)]
pub fn mat_mat_add(a: &Vec<Vec<Field>>, b: &Vec<Vec<Field>>) -> Vec<Vec<Field>>{
    a.iter().zip(b.iter()).map(|(ai,bi)| ai.iter().zip(bi.iter()).map(|(aij,bij)| (*aij + *bij) % MODULOS).collect()).collect()
}

#[cfg(test)]
mod mat_mat_add_tests {
    use super::*;
    #[test]
    fn mat_mat_add1() {
        let a = vec![vec![from_num(1.0),from_num(2.0)],
                     vec![from_num(3.0),from_num(4.0)],
                     vec![from_num(5.0),from_num(6.0)]];
        let b = vec![vec![from_num(1.0),from_num(1.0)],
                     vec![from_num(1.0),from_num(1.0)],
                     vec![from_num(1.0),from_num(1.0)]];
        let c = vec![vec![from_num(2.0),from_num(3.0)],
                     vec![from_num(4.0),from_num(5.0)],
                     vec![from_num(6.0),from_num(7.0)]];
        assert_eq!(c,mat_mat_add(&a,&b));
    }
}
#[allow(unused)]
pub fn mat_mat_sub(a: &Vec<Vec<Field>>, b: &Vec<Vec<Field>>) -> Vec<Vec<Field>>{
    a.iter().zip(b.iter()).map(|(ai,bi)| ai.iter().zip(bi.iter()).map(|(aij,bij)| subtract(*aij,*bij)).collect()).collect()
}

#[cfg(test)]
mod mat_mat_sub_tests {
    use super::*;
    #[test]
    fn mat_mat_sub1() {
        let a = vec![vec![from_num(1.0),from_num(2.0)],
                     vec![from_num(3.0),from_num(4.0)],
                     vec![from_num(5.0),from_num(6.0)]];
        let b = vec![vec![from_num(1.0),from_num(1.0)],
                     vec![from_num(1.0),from_num(1.0)],
                     vec![from_num(1.0),from_num(1.0)]];
        let c = vec![vec![from_num(0.0),from_num(1.0)],
                     vec![from_num(2.0),from_num(3.0)],
                     vec![from_num(4.0),from_num(5.0)]];
        assert_eq!(c,mat_mat_sub(&a,&b));
    }
}
#[allow(unused)]
pub fn vec_sca_mul(a: &Vec<Field>, b: Field) -> Vec<Field>{
    a.iter().map(|ai| multiply(*ai,b)).collect()
}

#[cfg(test)]
mod vec_sca_mul_tests {
    use super::*;
    #[test]
    fn vec_sca_mul1() {
        let a = vec![from_num(1.0),from_num(2.0),from_num(-4.3)];
        let b = from_num(-2.0);
        let c = vec![from_num(-2.0),from_num(-4.0),from_num(8.6)];
        assert_eq!(c,vec_sca_mul(&a,b));
    }
}
#[allow(unused)]
pub fn vec_sca_div(a: &Vec<Field>, b: Field) -> Vec<Field>{
    a.iter().map(|ai| divide(*ai,b)).collect()
}

#[cfg(test)]
mod vec_sca_div_tests {
    use super::*;
    #[test]
    fn vec_sca_div1() {
        let a = vec![from_num(1.0),from_num(2.0),from_num(-4.5)];
        let b = from_num(-2.0);
        let c = vec![from_num(-0.5),from_num(-1.0),from_num(2.25)];
        assert_eq!(c,vec_sca_div(&a,b));
    }
}
#[allow(unused)]
pub fn vec_vec_sub(a: &Vec<Field>, b: &Vec<Field>) -> Vec<Field>{
    a.iter().zip(b.iter()).map(|(ai,bi)| subtract(*ai,*bi)).collect()
}

#[cfg(test)]
mod vec_vec_sub_tests {
    use super::*;
    #[test]
    fn vec_vec_sub1() {
        let a = vec![from_num(1.0),from_num(2.0)];
        let b = vec![from_num(1.0),from_num(-1.0)];
        let c = vec![from_num(0.0),from_num(3.0)];
        assert_eq!(c,vec_vec_sub(&a,&b));
    }
}
#[allow(unused)]
pub fn mat_sca_mul(a: &Vec<Vec<Field>>, b: Field) -> Vec<Vec<Field>> {
    let mut res = vec![vec![from_num(0.0);a[0].len()];a.len()];
    for (i,ai) in a.iter().enumerate(){
        res[i] = ai.iter().map(|aij| multiply(*aij,b)).collect();
    }
    res
}

#[cfg(test)]
mod mat_sca_mul_tests {
    use super::*;
    #[test]
    fn mat_sca_mul1() {
        let a = vec![vec![from_num(1.0),from_num(2.0),from_num(-3.0)],
                     vec![from_num(0.0),from_num(-4.0),from_num(5.0)]];
        let b = vec![vec![from_num(2.0),from_num(4.0),from_num(-6.0)],
                     vec![from_num(0.0),from_num(-8.0),from_num(10.0)]];    
        assert_eq!(b,mat_sca_mul(&a,from_num(2.0)));
    }
}
#[allow(unused)]
pub fn mat_sca_div(a: &Vec<Vec<Field>>, b: Field) -> Vec<Vec<Field>> {
    let mut res = vec![vec![from_num(0.0);a[0].len()];a.len()];
    for (i,ai) in a.iter().enumerate(){
        res[i] = ai.iter().map(|aij| divide(*aij,b)).collect();
    }
    res
}

#[cfg(test)]
mod mat_sca_div_tests {
    use super::*;
    #[test]
    fn mat_sca_div1() {
        let a = vec![vec![from_num(1.0),from_num(2.0),from_num(-3.0)],
                     vec![from_num(0.0),from_num(-4.0),from_num(5.0)]];
        let b = vec![vec![from_num(2.0),from_num(4.0),from_num(-6.0)],
                     vec![from_num(0.0),from_num(-8.0),from_num(10.0)]];    
        assert_eq!(a,mat_sca_div(&b,from_num(2.0)));
    }
}
#[allow(unused)]
pub fn mat_vec_mul(a: &Vec<Vec<Field>>, x: &Vec<Field>) -> Vec<Field> {
    let mut b = vec![from_num(0.0);a.len()];
    for (i,ai) in a.iter().enumerate(){
        let mut sum = from_num(0.0);
        for (aij,xj) in ai.iter().zip(x.iter()) {
            sum = (sum + multiply(*aij,*xj)) % MODULOS;
        }
        b[i] = sum;
    }
    b
}

#[cfg(test)]
mod mat_vec_mul_tests {
    use super::*;
    #[test]
    fn mat_vec_mul1() {
        let a = vec![vec![from_num(1.0),from_num(0.0)],
                     vec![from_num(0.0),from_num(1.0)]];
        let x = vec![from_num(2.0),from_num(3.0)];
        let b = mat_vec_mul(&a,&x);
        assert_eq!(x,b);
    }
    #[test]
    fn mat_vec_mul2() {
        let a = vec![vec![from_num(1.0),from_num(1.0)],
                     vec![from_num(-1.0),from_num(1.0)]];
        let x = vec![from_num(2.0),from_num(3.0)];
        let res = vec![from_num(5.0),from_num(1.0)];
        let b = mat_vec_mul(&a,&x);
        assert_eq!(res,b);
    }
    #[test]
    fn mat_vec_mul3() {
        let a = vec![vec![from_num(1.0),from_num(1.0)],
                     vec![from_num(-1.0),from_num(1.0)],
                     vec![from_num(5.0),from_num(0.0)]];
        let x = vec![from_num(2.0),from_num(3.0)];
        let res = vec![from_num(5.0),from_num(1.0),from_num(10.0)];
        let b = mat_vec_mul(&a,&x);
        assert_eq!(res,b);
    }
}
#[allow(unused)]
pub fn transpose(a: &Vec<Vec<Field>>) -> Vec<Vec<Field>> {
    let mut a_t = vec![vec![from_num(0.0);a.len()];a[0].len()];
    for (i,a_t_i) in a_t.iter_mut().enumerate(){
        for (j,a_t_ij) in a_t_i.iter_mut().enumerate(){
            *a_t_ij = a[j][i];
        }
    }
    a_t
}

#[cfg(test)]
mod transpose_tests {
    use super::*;
    #[test]
    fn transpose1() {
        let a = vec![vec![from_num(1.0),from_num(0.0)],
                     vec![from_num(0.0),from_num(1.0)]];
        let a_t = vec![vec![from_num(1.0),from_num(0.0)],
                     vec![from_num(0.0),from_num(1.0)]];
        assert_eq!(a_t,transpose(&a));
    }
    #[test]
    fn transpose2() {
        let a = vec![vec![from_num(1.0),from_num(2.0)],
                     vec![from_num(3.0),from_num(4.0)]];
        let a_t = vec![vec![from_num(1.0),from_num(3.0)],
                     vec![from_num(2.0),from_num(4.0)]];
        assert_eq!(a_t,transpose(&a));
    }
    #[test]
    fn transpose3() {
        let a = vec![vec![from_num(1.0),from_num(0.0)],
                     vec![from_num(0.0),from_num(1.0)],
                     vec![from_num(2.0),from_num(4.0)]];
        let a_t = vec![vec![from_num(1.0),from_num(0.0),from_num(2.0)],
                      vec![from_num(0.0),from_num(1.0),from_num(4.0)]];
        assert_eq!(a_t,transpose(&a));
    }
}
#[allow(unused)]
pub fn mat_mat_mul(a: &Vec<Vec<Field>>, b: &Vec<Vec<Field>>) -> Vec<Vec<Field>>{
    let mut c = vec![vec![from_num(0.0);a.len()];b[0].len()];
    c.par_iter_mut().zip(transpose(b).par_iter()).for_each(|(ci,bi)| {
        *ci = mat_vec_mul(a,bi);
    });
    transpose(&c)
}

#[cfg(test)]
mod mat_mat_mul_tests {
    use super::*;
    #[test]
    fn mat_mat_mul1() {
        let a = vec![vec![from_num(1.0),from_num(0.0)],
                     vec![from_num(0.0),from_num(1.0)],
                     vec![from_num(1.0),from_num(1.0)]];
        let b = vec![vec![from_num(1.0),from_num(2.0)],
                     vec![from_num(3.0),from_num(4.0)]];
        let c = vec![vec![from_num(1.0),from_num(2.0)],
                     vec![from_num(3.0),from_num(4.0)],
                     vec![from_num(4.0),from_num(6.0)]];
        assert_eq!(c,mat_mat_mul(&a,&b));
    }
    #[test]
    fn mat_mat_mul2() {
        let a = vec![vec![from_num(1.0),from_num(0.0),from_num(2.0)],
                     vec![from_num(0.0),from_num(1.0),from_num(2.0)]];
        let b = vec![vec![from_num(1.0),from_num(2.0)],
                     vec![from_num(3.0),from_num(4.0)],
                     vec![from_num(5.0),from_num(6.0)]];
        let c = vec![vec![from_num(11.0),from_num(14.0)],
                     vec![from_num(13.0),from_num(16.0)]];
        assert_eq!(c,mat_mat_mul(&a,&b));
    }
}
#[allow(unused)]
pub fn sq_norm(a: &Vec<Field>) -> f64 {
    let mut n = 0.0;
    for ai in a.iter() {
        n += to_float(*ai).powf(2.0);
    }
    n
}

#[cfg(test)]
mod sq_norm_tests {
    use super::*;
    #[test]
    fn sq_norm1() {
        let a = vec![from_num(1.0),from_num(0.0)];
        let n = sq_norm(&a);
        assert_eq!(n,1.0);
    }
    #[test]
    fn sq_norm2() {
        let a = vec![from_num(2.0),from_num(2.0)];
        let n = sq_norm(&a);
        assert_eq!(n,8.0);
    }
    #[test]
    fn sq_norm3() {
        let a = vec![from_num(1.0),from_num(-2.0),from_num(4.0)];
        let n = sq_norm(&a);
        assert_eq!(n,21.0);
    }
}