#[cfg(test)]
use crate::probability::information_unit::InformationUnit;

#[test]
fn test_information_unit() {
    let bit: InformationUnit = InformationUnit::Bit(1.);
    let nat: InformationUnit = InformationUnit::Nat(1.);
    assert_eq!(bit.to_bits().to_float(), bit.to_float());
    assert_eq!(nat.to_nats().to_float(), nat.to_float());
    assert_eq!(bit.to_float() + bit.to_float(), (bit + bit).to_float());
    assert_eq!(nat.to_float() + nat.to_float(), (nat + nat).to_float());
    assert_eq!(bit.to_nats().to_bits().to_float(), bit.to_float());
    assert_eq!(nat.to_bits().to_nats().to_float(), nat.to_float());
    assert_eq!(nat.to_float(), 2f64.ln() * nat.to_bits().to_float());
}
