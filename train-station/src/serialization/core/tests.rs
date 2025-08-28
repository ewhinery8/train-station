//! Comprehensive roundtrip serialization tests
//!
//! This module contains test cases that verify roundtrip serialization and deserialization
//! functionality for both JSON and binary formats across structs of varying complexity.

use super::*;
use std::collections::HashMap;

/// Simple struct with basic primitive fields
#[derive(Debug, Clone, PartialEq)]
struct SimpleStruct {
    id: u32,
    name: String,
    active: bool,
    score: f64,
}

impl StructSerializable for SimpleStruct {
    fn to_serializer(&self) -> StructSerializer {
        StructSerializer::new()
            .field("id", &self.id)
            .field("name", &self.name)
            .field("active", &self.active)
            .field("score", &self.score)
    }

    fn from_deserializer(deserializer: &mut StructDeserializer) -> SerializationResult<Self> {
        let id = deserializer.field("id")?;
        let name = deserializer.field("name")?;
        let active = deserializer.field("active")?;
        let score = deserializer.field("score")?;

        Ok(SimpleStruct {
            id,
            name,
            active,
            score,
        })
    }
}

impl ToFieldValue for SimpleStruct {
    fn to_field_value(&self) -> FieldValue {
        let serializer = self.to_serializer();
        FieldValue::from_object(serializer.fields.into_iter().collect())
    }
}

impl FromFieldValue for SimpleStruct {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        match value {
            FieldValue::Object(fields) => {
                let mut deserializer = StructDeserializer { fields };
                Self::from_deserializer(&mut deserializer)
            }
            _ => Err(SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!(
                    "Expected Object for SimpleStruct, found {:?}",
                    value.type_name()
                ),
            }),
        }
    }
}

/// Medium complexity struct with collections
#[derive(Debug, Clone)]
struct MediumStruct {
    metadata: HashMap<String, String>,
    values: Vec<f32>,
    tags: Vec<String>,
    config: HashMap<String, String>,
}

impl PartialEq for MediumStruct {
    fn eq(&self, other: &Self) -> bool {
        self.metadata == other.metadata
            && self.values.len() == other.values.len()
            && self
                .values
                .iter()
                .zip(other.values.iter())
                .all(|(a, b)| (a - b).abs() < 1e-5)
            && self.tags == other.tags
            && self.config == other.config
    }
}

impl StructSerializable for MediumStruct {
    fn to_serializer(&self) -> StructSerializer {
        StructSerializer::new()
            .field("metadata", &self.metadata)
            .field("values", &self.values)
            .field("tags", &self.tags)
            .field("config", &self.config)
    }

    fn from_deserializer(deserializer: &mut StructDeserializer) -> SerializationResult<Self> {
        let metadata = deserializer.field("metadata")?;
        let values = deserializer.field("values")?;
        let tags = deserializer.field("tags")?;
        let config = deserializer.field("config")?;

        Ok(MediumStruct {
            metadata,
            values,
            tags,
            config,
        })
    }
}

impl ToFieldValue for MediumStruct {
    fn to_field_value(&self) -> FieldValue {
        let serializer = self.to_serializer();
        FieldValue::from_object(serializer.fields.into_iter().collect())
    }
}

impl FromFieldValue for MediumStruct {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        match value {
            FieldValue::Object(fields) => {
                let mut deserializer = StructDeserializer { fields };
                Self::from_deserializer(&mut deserializer)
            }
            _ => Err(SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!(
                    "Expected Object for MediumStruct, found {:?}",
                    value.type_name()
                ),
            }),
        }
    }
}

/// Complex nested struct with multiple levels of nesting
#[derive(Debug, Clone)]
struct ComplexStruct {
    simple_data: SimpleStruct,
    medium_data: MediumStruct,
    nested_structs: Vec<SimpleStruct>,
    nested_maps: HashMap<String, String>,
    matrix_data: Vec<f64>,
    matrix_rows: usize,
    matrix_cols: usize,
    optional_data: Option<String>,
    byte_data: Vec<u8>,
}

impl PartialEq for ComplexStruct {
    fn eq(&self, other: &Self) -> bool {
        // Check simple_data with floating-point tolerance
        self.simple_data.id == other.simple_data.id
            && self.simple_data.name == other.simple_data.name
            && self.simple_data.active == other.simple_data.active
            && (self.simple_data.score - other.simple_data.score).abs() < 1e-5
            && self.medium_data == other.medium_data
            && self.nested_structs.len() == other.nested_structs.len()
            && self
                .nested_structs
                .iter()
                .zip(other.nested_structs.iter())
                .all(|(a, b)| {
                    a.id == b.id
                        && a.name == b.name
                        && a.active == b.active
                        && (a.score - b.score).abs() < 1e-5
                })
            && self.nested_maps == other.nested_maps
            && self.matrix_rows == other.matrix_rows
            && self.matrix_cols == other.matrix_cols
            && self.matrix_data.len() == other.matrix_data.len()
            && self
                .matrix_data
                .iter()
                .zip(other.matrix_data.iter())
                .all(|(a, b)| (a - b).abs() < 1e-5)
            && self.optional_data == other.optional_data
            && self.byte_data == other.byte_data
    }
}

impl StructSerializable for ComplexStruct {
    fn to_serializer(&self) -> StructSerializer {
        StructSerializer::new()
            .field("simple_data", &self.simple_data)
            .field("medium_data", &self.medium_data)
            .field("nested_structs", &self.nested_structs)
            .field("nested_maps", &self.nested_maps)
            .field("matrix_data", &self.matrix_data)
            .field("matrix_rows", &self.matrix_rows)
            .field("matrix_cols", &self.matrix_cols)
            .field("optional_data", &self.optional_data)
            .field("byte_data", &self.byte_data)
    }

    fn from_deserializer(deserializer: &mut StructDeserializer) -> SerializationResult<Self> {
        let simple_data = deserializer.field("simple_data")?;
        let medium_data = deserializer.field("medium_data")?;
        let nested_structs = deserializer.field("nested_structs")?;
        let nested_maps = deserializer.field("nested_maps")?;
        let matrix_data = deserializer.field("matrix_data")?;
        let matrix_rows = deserializer.field("matrix_rows")?;
        let matrix_cols = deserializer.field("matrix_cols")?;
        let optional_data = deserializer.field("optional_data")?;
        let byte_data = deserializer.field("byte_data")?;

        Ok(ComplexStruct {
            simple_data,
            medium_data,
            nested_structs,
            nested_maps,
            matrix_data,
            matrix_rows,
            matrix_cols,
            optional_data,
            byte_data,
        })
    }
}

impl ToFieldValue for ComplexStruct {
    fn to_field_value(&self) -> FieldValue {
        let serializer = self.to_serializer();
        FieldValue::from_object(serializer.fields.into_iter().collect())
    }
}

impl FromFieldValue for ComplexStruct {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        match value {
            FieldValue::Object(fields) => {
                let mut deserializer = StructDeserializer { fields };
                Self::from_deserializer(&mut deserializer)
            }
            _ => Err(SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!(
                    "Expected Object for ComplexStruct, found {:?}",
                    value.type_name()
                ),
            }),
        }
    }
}

#[cfg(test)]
/// Test enums for comprehensive enum serialization testing
mod test_enums {
    use super::*;
    use std::collections::HashMap;

    /// Standard enum with unit, tuple, and struct variants
    #[derive(Debug, PartialEq, Clone)]
    pub enum StandardEnum {
        /// Unit variant - no associated data
        Unit,
        /// Tuple variant with multiple fields
        Tuple(i32, String, bool),
        /// Struct variant with named fields
        Struct { id: u32, name: String, active: bool },
    }

    impl ToFieldValue for StandardEnum {
        fn to_field_value(&self) -> FieldValue {
            match self {
                StandardEnum::Unit => FieldValue::from_enum_unit("Unit".to_string()),
                StandardEnum::Tuple(a, b, c) => FieldValue::from_enum_tuple(
                    "Tuple".to_string(),
                    vec![a.to_field_value(), b.to_field_value(), c.to_field_value()],
                ),
                StandardEnum::Struct { id, name, active } => {
                    let mut fields = HashMap::new();
                    fields.insert("id".to_string(), id.to_field_value());
                    fields.insert("name".to_string(), name.to_field_value());
                    fields.insert("active".to_string(), active.to_field_value());
                    FieldValue::from_enum_struct("Struct".to_string(), fields)
                }
            }
        }
    }

    impl FromFieldValue for StandardEnum {
        fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
            let (variant, data) =
                value
                    .as_enum()
                    .map_err(|_| SerializationError::ValidationFailed {
                        field: field_name.to_string(),
                        message: "Expected enum value".to_string(),
                    })?;

            match variant {
                "Unit" => Ok(StandardEnum::Unit),
                "Tuple" => {
                    let data = data.ok_or_else(|| SerializationError::ValidationFailed {
                        field: field_name.to_string(),
                        message: "Tuple variant missing data".to_string(),
                    })?;
                    let array = data.as_array()?;
                    if array.len() != 3 {
                        return Err(SerializationError::ValidationFailed {
                            field: field_name.to_string(),
                            message: "Tuple variant requires exactly 3 elements".to_string(),
                        });
                    }
                    let a = i32::from_field_value(array[0].clone(), "tuple.0")?;
                    let b = String::from_field_value(array[1].clone(), "tuple.1")?;
                    let c = bool::from_field_value(array[2].clone(), "tuple.2")?;
                    Ok(StandardEnum::Tuple(a, b, c))
                }
                "Struct" => {
                    let data = data.ok_or_else(|| SerializationError::ValidationFailed {
                        field: field_name.to_string(),
                        message: "Struct variant missing data".to_string(),
                    })?;
                    let object = data.as_object()?;
                    let id = u32::from_field_value(
                        object.get("id").cloned().unwrap_or(FieldValue::U32(0)),
                        "struct.id",
                    )?;
                    let name = String::from_field_value(
                        object
                            .get("name")
                            .cloned()
                            .unwrap_or(FieldValue::String("".to_string())),
                        "struct.name",
                    )?;
                    let active = bool::from_field_value(
                        object
                            .get("active")
                            .cloned()
                            .unwrap_or(FieldValue::Bool(false)),
                        "struct.active",
                    )?;
                    Ok(StandardEnum::Struct { id, name, active })
                }
                _ => Err(SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: format!("Unknown enum variant: {}", variant),
                }),
            }
        }
    }

    /// Simple enum with only unit variants
    #[derive(Debug, PartialEq, Clone)]
    pub enum SimpleEnum {
        Red,
        Green,
        Blue,
    }

    impl ToFieldValue for SimpleEnum {
        fn to_field_value(&self) -> FieldValue {
            match self {
                SimpleEnum::Red => FieldValue::from_enum_unit("Red".to_string()),
                SimpleEnum::Green => FieldValue::from_enum_unit("Green".to_string()),
                SimpleEnum::Blue => FieldValue::from_enum_unit("Blue".to_string()),
            }
        }
    }

    impl FromFieldValue for SimpleEnum {
        fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
            let (variant, data) =
                value
                    .as_enum()
                    .map_err(|_| SerializationError::ValidationFailed {
                        field: field_name.to_string(),
                        message: "Expected enum value".to_string(),
                    })?;

            if data.is_some() {
                return Err(SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: "Unit variant should not have data".to_string(),
                });
            }

            match variant {
                "Red" => Ok(SimpleEnum::Red),
                "Green" => Ok(SimpleEnum::Green),
                "Blue" => Ok(SimpleEnum::Blue),
                _ => Err(SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: format!("Unknown color variant: {}", variant),
                }),
            }
        }
    }

    /// Complex nested enum for testing complex scenarios
    #[derive(Debug, PartialEq, Clone)]
    pub enum NestedEnum {
        Simple(SimpleEnum),
        Complex {
            standard: StandardEnum,
            values: Vec<i32>,
            metadata: HashMap<String, String>,
        },
    }

    impl ToFieldValue for NestedEnum {
        fn to_field_value(&self) -> FieldValue {
            match self {
                NestedEnum::Simple(simple) => {
                    FieldValue::from_enum_tuple("Simple".to_string(), vec![simple.to_field_value()])
                }
                NestedEnum::Complex {
                    standard,
                    values,
                    metadata,
                } => {
                    let mut fields = HashMap::new();
                    fields.insert("standard".to_string(), standard.to_field_value());
                    fields.insert("values".to_string(), values.to_field_value());
                    fields.insert("metadata".to_string(), metadata.to_field_value());
                    FieldValue::from_enum_struct("Complex".to_string(), fields)
                }
            }
        }
    }

    impl FromFieldValue for NestedEnum {
        fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
            let (variant, data) =
                value
                    .as_enum()
                    .map_err(|_| SerializationError::ValidationFailed {
                        field: field_name.to_string(),
                        message: "Expected enum value".to_string(),
                    })?;

            match variant {
                "Simple" => {
                    let data = data.ok_or_else(|| SerializationError::ValidationFailed {
                        field: field_name.to_string(),
                        message: "Simple variant missing data".to_string(),
                    })?;
                    let array = data.as_array()?;
                    if array.len() != 1 {
                        return Err(SerializationError::ValidationFailed {
                            field: field_name.to_string(),
                            message: "Simple variant requires exactly 1 element".to_string(),
                        });
                    }
                    let simple = SimpleEnum::from_field_value(array[0].clone(), "nested.simple")?;
                    Ok(NestedEnum::Simple(simple))
                }
                "Complex" => {
                    let data = data.ok_or_else(|| SerializationError::ValidationFailed {
                        field: field_name.to_string(),
                        message: "Complex variant missing data".to_string(),
                    })?;
                    let object = data.as_object()?;
                    let standard = StandardEnum::from_field_value(
                        object
                            .get("standard")
                            .cloned()
                            .unwrap_or(FieldValue::from_enum_unit("Unit".to_string())),
                        "nested.standard",
                    )?;
                    let values = Vec::<i32>::from_field_value(
                        object
                            .get("values")
                            .cloned()
                            .unwrap_or(FieldValue::Array(vec![])),
                        "nested.values",
                    )?;
                    let metadata = HashMap::<String, String>::from_field_value(
                        object
                            .get("metadata")
                            .cloned()
                            .unwrap_or(FieldValue::Object(HashMap::new())),
                        "nested.metadata",
                    )?;
                    Ok(NestedEnum::Complex {
                        standard,
                        values,
                        metadata,
                    })
                }
                _ => Err(SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: format!("Unknown nested variant: {}", variant),
                }),
            }
        }
    }
}

mod enum_tests {
    use super::test_enums::*;
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_simple_enum_field_value() {
        let red = SimpleEnum::Red;
        let field_value = red.to_field_value();

        let (variant, data) = field_value.as_enum().unwrap();
        assert_eq!(variant, "Red");
        assert!(data.is_none());

        let reconstructed = SimpleEnum::from_field_value(field_value, "test").unwrap();
        assert_eq!(red, reconstructed);
    }

    #[test]
    fn test_standard_enum_unit_variant() {
        let unit = StandardEnum::Unit;
        let field_value = unit.to_field_value();

        let (variant, data) = field_value.as_enum().unwrap();
        assert_eq!(variant, "Unit");
        assert!(data.is_none());

        let reconstructed = StandardEnum::from_field_value(field_value, "test").unwrap();
        assert_eq!(unit, reconstructed);
    }

    #[test]
    fn test_standard_enum_tuple_variant() {
        let tuple = StandardEnum::Tuple(42, "hello".to_string(), true);
        let field_value = tuple.to_field_value();

        let (variant, data) = field_value.as_enum().unwrap();
        assert_eq!(variant, "Tuple");
        assert!(data.is_some());

        let data_array = data.unwrap().as_array().unwrap();
        assert_eq!(data_array.len(), 3);
        assert_eq!(data_array[0].as_i32().unwrap(), 42);
        assert_eq!(data_array[1].as_string().unwrap(), "hello");
        assert!(data_array[2].as_bool().unwrap());

        let reconstructed = StandardEnum::from_field_value(field_value, "test").unwrap();
        assert_eq!(tuple, reconstructed);
    }

    #[test]
    fn test_standard_enum_struct_variant() {
        let struct_variant = StandardEnum::Struct {
            id: 123,
            name: "test".to_string(),
            active: false,
        };
        let field_value = struct_variant.to_field_value();

        let (variant, data) = field_value.as_enum().unwrap();
        assert_eq!(variant, "Struct");
        assert!(data.is_some());

        let data_object = data.unwrap().as_object().unwrap();
        assert_eq!(data_object.len(), 3);
        assert_eq!(data_object.get("id").unwrap().as_u32().unwrap(), 123);
        assert_eq!(
            data_object.get("name").unwrap().as_string().unwrap(),
            "test"
        );
        assert!(!data_object.get("active").unwrap().as_bool().unwrap());

        let reconstructed = StandardEnum::from_field_value(field_value, "test").unwrap();
        assert_eq!(struct_variant, reconstructed);
    }

    #[test]
    fn test_nested_enum_simple_variant() {
        let nested = NestedEnum::Simple(SimpleEnum::Green);
        let field_value = nested.to_field_value();

        let (variant, data) = field_value.as_enum().unwrap();
        assert_eq!(variant, "Simple");
        assert!(data.is_some());

        let reconstructed = NestedEnum::from_field_value(field_value, "test").unwrap();
        assert_eq!(nested, reconstructed);
    }

    #[test]
    fn test_nested_enum_complex_variant() {
        let mut metadata = HashMap::new();
        metadata.insert("key1".to_string(), "value1".to_string());
        metadata.insert("key2".to_string(), "value2".to_string());

        let nested = NestedEnum::Complex {
            standard: StandardEnum::Tuple(10, "nested".to_string(), false),
            values: vec![1, 2, 3, 4, 5],
            metadata,
        };

        let field_value = nested.to_field_value();

        let (variant, data) = field_value.as_enum().unwrap();
        assert_eq!(variant, "Complex");
        assert!(data.is_some());

        let reconstructed = NestedEnum::from_field_value(field_value, "test").unwrap();
        assert_eq!(nested, reconstructed);
    }

    #[test]
    fn test_enum_json_serialization() {
        let test_cases = vec![
            StandardEnum::Unit,
            StandardEnum::Tuple(42, "test".to_string(), true),
            StandardEnum::Struct {
                id: 999,
                name: "serialization".to_string(),
                active: true,
            },
        ];

        for test_case in test_cases {
            let serializer = StructSerializer::new().field("enum_field", &test_case);

            let json = serializer.to_json().unwrap();
            let mut deserializer = StructDeserializer::from_json(&json).unwrap();
            let reconstructed: StandardEnum = deserializer.field("enum_field").unwrap();

            assert_eq!(test_case, reconstructed);
        }
    }

    #[test]
    fn test_enum_binary_serialization() {
        let test_cases = vec![
            StandardEnum::Unit,
            StandardEnum::Tuple(42, "test".to_string(), true),
            StandardEnum::Struct {
                id: 999,
                name: "serialization".to_string(),
                active: true,
            },
        ];

        for test_case in test_cases {
            let serializer = StructSerializer::new().field("enum_field", &test_case);

            let binary = serializer.to_binary().unwrap();
            let mut deserializer = StructDeserializer::from_binary(&binary).unwrap();
            let reconstructed: StandardEnum = deserializer.field("enum_field").unwrap();

            assert_eq!(test_case, reconstructed);
        }
    }

    #[test]
    fn test_simple_enum_all_variants_json() {
        let variants = vec![SimpleEnum::Red, SimpleEnum::Green, SimpleEnum::Blue];

        for variant in variants {
            let serializer = StructSerializer::new().field("color", &variant);

            let json = serializer.to_json().unwrap();
            let mut deserializer = StructDeserializer::from_json(&json).unwrap();
            let reconstructed: SimpleEnum = deserializer.field("color").unwrap();

            assert_eq!(variant, reconstructed);
        }
    }

    #[test]
    fn test_simple_enum_all_variants_binary() {
        let variants = vec![SimpleEnum::Red, SimpleEnum::Green, SimpleEnum::Blue];

        for variant in variants {
            let serializer = StructSerializer::new().field("color", &variant);

            let binary = serializer.to_binary().unwrap();
            let mut deserializer = StructDeserializer::from_binary(&binary).unwrap();
            let reconstructed: SimpleEnum = deserializer.field("color").unwrap();

            assert_eq!(variant, reconstructed);
        }
    }

    #[test]
    fn test_nested_enum_complex_serialization() {
        let mut metadata = HashMap::new();
        metadata.insert("author".to_string(), "test".to_string());
        metadata.insert("version".to_string(), "1.0".to_string());

        let complex_nested = NestedEnum::Complex {
            standard: StandardEnum::Struct {
                id: 42,
                name: "complex_test".to_string(),
                active: true,
            },
            values: vec![10, 20, 30, 40, 50],
            metadata,
        };

        // Test JSON serialization
        let serializer = StructSerializer::new().field("nested", &complex_nested);
        let json = serializer.to_json().unwrap();

        let mut deserializer = StructDeserializer::from_json(&json).unwrap();
        let reconstructed_json: NestedEnum = deserializer.field("nested").unwrap();
        assert_eq!(complex_nested, reconstructed_json);

        // Test Binary serialization
        let serializer = StructSerializer::new().field("nested", &complex_nested);
        let binary = serializer.to_binary().unwrap();

        let mut deserializer = StructDeserializer::from_binary(&binary).unwrap();
        let reconstructed_binary: NestedEnum = deserializer.field("nested").unwrap();
        assert_eq!(complex_nested, reconstructed_binary);
    }

    #[test]
    fn test_enum_error_handling() {
        // Test unknown variant
        let invalid_enum = FieldValue::from_enum_unit("InvalidVariant".to_string());
        let result = StandardEnum::from_field_value(invalid_enum, "test");
        assert!(result.is_err());

        // Test missing data for tuple variant
        let missing_data = FieldValue::from_enum_unit("Tuple".to_string());
        let result = StandardEnum::from_field_value(missing_data, "test");
        assert!(result.is_err());

        // Test wrong data type
        let wrong_data = FieldValue::from_enum_tuple("Struct".to_string(), vec![]);
        let result = StandardEnum::from_field_value(wrong_data, "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_enum_edge_cases() {
        // Test empty strings and special characters
        let special_tuple = StandardEnum::Tuple(0, "".to_string(), false);
        let serializer = StructSerializer::new().field("special", &special_tuple);

        let json = serializer.to_json().unwrap();
        let mut deserializer = StructDeserializer::from_json(&json).unwrap();
        let reconstructed: StandardEnum = deserializer.field("special").unwrap();
        assert_eq!(special_tuple, reconstructed);

        // Test large values
        let large_tuple = StandardEnum::Tuple(i32::MAX, "x".repeat(1000), true);
        let serializer = StructSerializer::new().field("large", &large_tuple);

        let binary = serializer.to_binary().unwrap();
        let mut deserializer = StructDeserializer::from_binary(&binary).unwrap();
        let reconstructed: StandardEnum = deserializer.field("large").unwrap();
        assert_eq!(large_tuple, reconstructed);
    }

    #[test]
    fn test_enum_with_optional_fields() {
        let optional_enum = StandardEnum::Struct {
            id: 42,
            name: "optional_test".to_string(),
            active: true,
        };

        let serializer = StructSerializer::new()
            .field("required_enum", &optional_enum)
            .field("optional_enum", &optional_enum);

        // Test that enum fields work in serialization context
        let json = serializer.to_json().unwrap();
        let mut deserializer = StructDeserializer::from_json(&json).unwrap();

        let required: StandardEnum = deserializer.field("required_enum").unwrap();
        let optional: StandardEnum = deserializer.field("optional_enum").unwrap();

        assert_eq!(required, optional_enum);
        assert_eq!(optional, optional_enum);
    }

    #[test]
    fn test_multiple_enums_in_struct() {
        let mut metadata = HashMap::new();
        metadata.insert("tag".to_string(), "multi_enum".to_string());

        let multi_enum_struct = StructSerializer::new()
            .field("simple", &SimpleEnum::Blue)
            .field("standard", &StandardEnum::Unit)
            .field("nested", &NestedEnum::Simple(SimpleEnum::Red))
            .field(
                "complex_nested",
                &NestedEnum::Complex {
                    standard: StandardEnum::Tuple(100, "multi".to_string(), false),
                    values: vec![1, 1, 2, 3, 5, 8],
                    metadata,
                },
            );

        // Test JSON
        let json = multi_enum_struct.to_json().unwrap();
        let mut json_deserializer = StructDeserializer::from_json(&json).unwrap();

        let simple: SimpleEnum = json_deserializer.field("simple").unwrap();
        let standard: StandardEnum = json_deserializer.field("standard").unwrap();
        let nested: NestedEnum = json_deserializer.field("nested").unwrap();
        let complex_nested: NestedEnum = json_deserializer.field("complex_nested").unwrap();

        assert_eq!(simple, SimpleEnum::Blue);
        assert_eq!(standard, StandardEnum::Unit);
        assert_eq!(nested, NestedEnum::Simple(SimpleEnum::Red));
        assert!(matches!(complex_nested, NestedEnum::Complex { .. }));

        // Test Binary - recreate the struct since multi_enum_struct was moved
        let mut metadata = HashMap::new();
        metadata.insert("tag".to_string(), "multi_enum".to_string());

        let multi_enum_struct_binary = StructSerializer::new()
            .field("simple", &SimpleEnum::Blue)
            .field("standard", &StandardEnum::Unit)
            .field("nested", &NestedEnum::Simple(SimpleEnum::Red))
            .field(
                "complex_nested",
                &NestedEnum::Complex {
                    standard: StandardEnum::Tuple(100, "multi".to_string(), false),
                    values: vec![1, 1, 2, 3, 5, 8],
                    metadata,
                },
            );

        let binary = multi_enum_struct_binary.to_binary().unwrap();
        let mut binary_deserializer = StructDeserializer::from_binary(&binary).unwrap();

        let simple: SimpleEnum = binary_deserializer.field("simple").unwrap();
        let standard: StandardEnum = binary_deserializer.field("standard").unwrap();
        let nested: NestedEnum = binary_deserializer.field("nested").unwrap();
        let complex_nested: NestedEnum = binary_deserializer.field("complex_nested").unwrap();

        assert_eq!(simple, SimpleEnum::Blue);
        assert_eq!(standard, StandardEnum::Unit);
        assert_eq!(nested, NestedEnum::Simple(SimpleEnum::Red));
        assert!(matches!(complex_nested, NestedEnum::Complex { .. }));
    }
}

mod roundtrip_tests {
    use super::*;

    #[test]
    fn test_simple_struct_roundtrip() {
        let original = SimpleStruct {
            id: 42u32,
            name: "test_struct".to_string(),
            active: true,
            score: 98.5,
        };

        // Test JSON roundtrip
        let json_data = original.to_json().unwrap();
        let json_deserialized = SimpleStruct::from_json(&json_data).unwrap();

        assert_eq!(original, json_deserialized);
        println!("Simple struct JSON: {}", json_data);

        // Test Binary roundtrip
        let binary_data = original.to_binary().unwrap();
        let binary_deserialized = SimpleStruct::from_binary(&binary_data).unwrap();

        assert_eq!(original, binary_deserialized);
        println!("Simple struct binary size: {} bytes", binary_data.len());
    }

    #[test]
    fn test_medium_struct_roundtrip() {
        let mut metadata = HashMap::new();
        metadata.insert("version".to_string(), "1.0".to_string());
        metadata.insert("author".to_string(), "test_user".to_string());

        let mut config = HashMap::new();
        config.insert("debug".to_string(), "true".to_string());
        config.insert("max_iterations".to_string(), "1000".to_string());

        let original = MediumStruct {
            metadata,
            values: vec![1.1, 2.2, 3.3, 4.4, 5.5],
            tags: vec![
                "important".to_string(),
                "test".to_string(),
                "data".to_string(),
            ],
            config,
        };

        // Test JSON roundtrip
        let json_data = original.to_json().unwrap();
        let json_deserialized = MediumStruct::from_json(&json_data).unwrap();

        assert_eq!(original, json_deserialized);
        println!("Medium struct JSON: {}", json_data);

        // Test Binary roundtrip
        let binary_data = original.to_binary().unwrap();
        let binary_deserialized = MediumStruct::from_binary(&binary_data).unwrap();

        assert_eq!(original, binary_deserialized);
        println!("Medium struct binary size: {} bytes", binary_data.len());
    }

    #[test]
    fn test_complex_struct_roundtrip() {
        let simple_data = SimpleStruct {
            id: 123u32,
            name: "nested_simple".to_string(),
            active: false,
            score: 87.3,
        };

        let mut metadata = HashMap::new();
        metadata.insert("experiment".to_string(), "complex_test".to_string());
        metadata.insert("dataset".to_string(), "synthetic".to_string());

        let mut config = HashMap::new();
        config.insert("learning_rate".to_string(), "0.001".to_string());
        config.insert("batch_size".to_string(), "32".to_string());

        let medium_data = MediumStruct {
            metadata,
            values: vec![10.1, 20.2, 30.3],
            tags: vec!["ml".to_string(), "training".to_string()],
            config,
        };

        let nested_structs = vec![
            SimpleStruct {
                id: 1u32,
                name: "first".to_string(),
                active: true,
                score: 95.0,
            },
            SimpleStruct {
                id: 2u32,
                name: "second".to_string(),
                active: false,
                score: 82.5,
            },
        ];

        let mut nested_maps = HashMap::new();
        nested_maps.insert("key1".to_string(), "value1".to_string());
        nested_maps.insert("key2".to_string(), "value2".to_string());

        let original = ComplexStruct {
            simple_data,
            medium_data,
            nested_structs,
            nested_maps,
            matrix_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            matrix_rows: 3usize,
            matrix_cols: 3usize,
            optional_data: Some("optional_value".to_string()),
            byte_data: vec![0x01, 0x02, 0x03, 0xFF, 0xAB],
        };

        // Test JSON roundtrip
        let json_data = original.to_json().unwrap();
        let json_deserialized = ComplexStruct::from_json(&json_data).unwrap();

        assert!(original == json_deserialized, "JSON roundtrip failed");
        println!("Complex struct JSON: {}", json_data);

        // Test Binary roundtrip
        let binary_data = original.to_binary().unwrap();
        let binary_deserialized = ComplexStruct::from_binary(&binary_data).unwrap();

        assert!(original == binary_deserialized, "Binary roundtrip failed");
        println!("Complex struct binary size: {} bytes", binary_data.len());
    }
}

/// File I/O roundtrip tests
#[cfg(test)]
mod file_io_tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    /// Test 1: Simple struct JSON file roundtrip
    ///
    /// Verifies that a simple struct can be saved to and loaded from a JSON file correctly.
    #[test]
    fn test_simple_struct_json_file_roundtrip() {
        let original = SimpleStruct {
            id: 42,
            name: "test_simple".to_string(),
            active: true,
            score: 98.5,
        };

        let test_file = "test_simple_struct.json";

        // Save to file
        original.save_json(test_file).unwrap();

        // Verify file exists and has content
        assert!(Path::new(test_file).exists());
        let file_content = fs::read_to_string(test_file).unwrap();
        assert!(!file_content.is_empty());

        // Load from file
        let loaded = SimpleStruct::load_json(test_file).unwrap();

        // Verify data integrity
        assert_eq!(original, loaded);

        // Clean up
        fs::remove_file(test_file).unwrap();
    }

    /// Test 2: Medium struct with collections JSON file roundtrip
    ///
    /// Verifies that a struct with vectors, maps, and complex types can be saved to and loaded from a JSON file correctly.
    #[test]
    fn test_medium_struct_json_file_roundtrip() {
        let mut metadata = HashMap::new();
        metadata.insert("author".to_string(), "test_user".to_string());
        metadata.insert("version".to_string(), "1.0.0".to_string());
        metadata.insert(
            "description".to_string(),
            "medium complexity test".to_string(),
        );

        let mut config = HashMap::new();
        config.insert("debug_mode".to_string(), "true".to_string());
        config.insert("max_iterations".to_string(), "1000".to_string());

        let original = MediumStruct {
            metadata,
            values: vec![1.5, 2.7, 3.25, -0.5, 100.0],
            tags: vec![
                "machine_learning".to_string(),
                "neural_networks".to_string(),
                "optimization".to_string(),
            ],
            config,
        };

        let test_file = "test_medium_struct.json";

        // Save to file
        original.save_json(test_file).unwrap();

        // Verify file exists and has content
        assert!(Path::new(test_file).exists());
        let file_content = fs::read_to_string(test_file).unwrap();
        assert!(!file_content.is_empty());
        assert!(file_content.contains("machine_learning")); // Spot check content
        assert!(file_content.contains("3.25")); // Spot check float precision

        // Load from file
        let loaded = MediumStruct::load_json(test_file).unwrap();

        // Verify data integrity
        assert_eq!(original, loaded);

        // Verify specific fields
        assert_eq!(loaded.values.len(), 5);
        assert_eq!(loaded.tags.len(), 3);
        assert_eq!(loaded.metadata.get("author").unwrap(), "test_user");
        assert!((loaded.values[2] - 3.25).abs() < f32::EPSILON);

        // Clean up
        fs::remove_file(test_file).unwrap();
    }

    /// Test 3: Complex nested struct with enums JSON file roundtrip
    ///
    /// Verifies that a complex nested struct with enums, optional values, and deep nesting
    /// can be saved to and loaded from a JSON file correctly.
    #[test]
    fn test_complex_nested_struct_with_enums_json_file_roundtrip() {
        // Create nested simple structs
        let nested_structs = vec![
            SimpleStruct {
                id: 1,
                name: "nested_1".to_string(),
                active: true,
                score: 85.5,
            },
            SimpleStruct {
                id: 2,
                name: "nested_2".to_string(),
                active: false,
                score: 92.3,
            },
        ];

        // Create nested maps
        let mut nested_maps = HashMap::new();
        nested_maps.insert("category_a".to_string(), "value_a".to_string());
        nested_maps.insert("category_b".to_string(), "value_b".to_string());

        // Create medium struct for nesting
        let mut medium_metadata = HashMap::new();
        medium_metadata.insert("subsystem".to_string(), "test_subsystem".to_string());
        medium_metadata.insert("level".to_string(), "deep".to_string());

        let mut medium_config = HashMap::new();
        medium_config.insert("nested_config".to_string(), "enabled".to_string());

        let medium_data = MediumStruct {
            metadata: medium_metadata,
            values: vec![10.1, 20.2, 30.3],
            tags: vec!["nested".to_string(), "complex".to_string()],
            config: medium_config,
        };

        // Create original complex struct
        let original = ComplexStruct {
            simple_data: SimpleStruct {
                id: 100,
                name: "main_simple".to_string(),
                active: true,
                score: 95.7,
            },
            medium_data,
            nested_structs,
            nested_maps,
            matrix_data: vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
            matrix_rows: 3,
            matrix_cols: 3,
            optional_data: Some("complex_optional_value".to_string()),
            byte_data: vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE],
        };

        let test_file = "test_complex_nested_struct.json";

        // Save to file
        original.save_json(test_file).unwrap();

        // Verify file exists and has substantial content
        assert!(Path::new(test_file).exists());
        let file_content = fs::read_to_string(test_file).unwrap();
        assert!(!file_content.is_empty());
        assert!(file_content.len() > 500); // Should be substantial JSON

        // Spot check for nested content
        assert!(file_content.contains("main_simple"));
        assert!(file_content.contains("nested_1"));
        assert!(file_content.contains("test_subsystem"));
        assert!(file_content.contains("complex_optional_value"));
        assert!(file_content.contains("deadbeefcafe")); // Byte arrays serialized as hex
        assert!(file_content.contains("3.3")); // Matrix data

        // Load from file
        let loaded = ComplexStruct::load_json(test_file).unwrap();

        // Verify data integrity
        assert_eq!(original, loaded);

        // Verify specific nested fields
        assert_eq!(loaded.simple_data.name, "main_simple");
        assert_eq!(loaded.nested_structs.len(), 2);
        assert_eq!(loaded.nested_structs[0].name, "nested_1");
        assert_eq!(loaded.nested_structs[1].name, "nested_2");
        assert_eq!(loaded.medium_data.tags.len(), 2);
        assert_eq!(loaded.medium_data.tags[0], "nested");
        assert_eq!(loaded.nested_maps.get("category_a").unwrap(), "value_a");
        assert_eq!(loaded.matrix_data.len(), 9);
        assert_eq!(loaded.matrix_data[2], 3.3); // Exact precision with JSON fix
        assert_eq!(
            loaded.optional_data.as_ref().unwrap(),
            "complex_optional_value"
        );
        assert_eq!(loaded.byte_data, vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE]);

        // Clean up
        fs::remove_file(test_file).unwrap();
    }

    /// Test 1: Simple struct binary file roundtrip
    ///
    /// Verifies that a simple struct can be saved to and loaded from a binary file correctly.
    #[test]
    fn test_simple_struct_binary_file_roundtrip() {
        let original = SimpleStruct {
            id: 42,
            name: "test_simple_binary".to_string(),
            active: true,
            score: 98.5,
        };

        let test_file = "test_simple_struct.bin";

        // Save to file
        original.save_binary(test_file).unwrap();

        // Verify file exists and has content
        assert!(Path::new(test_file).exists());
        let file_metadata = fs::metadata(test_file).unwrap();
        assert!(file_metadata.len() > 0);

        // Load from file
        let loaded = SimpleStruct::load_binary(test_file).unwrap();

        // Verify data integrity
        assert_eq!(original, loaded);

        // Clean up
        fs::remove_file(test_file).unwrap();
    }

    /// Test 2: Medium struct with collections binary file roundtrip
    ///
    /// Verifies that a struct with vectors, maps, and complex types can be saved to and loaded from a binary file correctly.
    #[test]
    fn test_medium_struct_binary_file_roundtrip() {
        let mut metadata = HashMap::new();
        metadata.insert("author".to_string(), "binary_test_user".to_string());
        metadata.insert("version".to_string(), "2.0.0".to_string());
        metadata.insert(
            "description".to_string(),
            "medium complexity binary test".to_string(),
        );

        let mut config = HashMap::new();
        config.insert("binary_mode".to_string(), "true".to_string());
        config.insert("max_iterations".to_string(), "2000".to_string());

        let original = MediumStruct {
            metadata,
            values: vec![1.5, 2.7, 3.25, -0.5, 100.0, 0.001],
            tags: vec![
                "binary_serialization".to_string(),
                "performance_test".to_string(),
                "data_integrity".to_string(),
            ],
            config,
        };

        let test_file = "test_medium_struct.bin";

        // Save to file
        original.save_binary(test_file).unwrap();

        // Verify file exists and has content
        assert!(Path::new(test_file).exists());
        let file_metadata = fs::metadata(test_file).unwrap();
        assert!(file_metadata.len() > 50); // Binary should be more compact than JSON

        // Binary files should be smaller than equivalent JSON
        // (This is just a rough check - actual size depends on data)
        assert!(file_metadata.len() < 1000);

        // Load from file
        let loaded = MediumStruct::load_binary(test_file).unwrap();

        // Verify data integrity
        assert_eq!(original, loaded);

        // Verify specific fields with exact precision (binary preserves full precision)
        assert_eq!(loaded.values.len(), 6);
        assert_eq!(loaded.tags.len(), 3);
        assert_eq!(loaded.metadata.get("author").unwrap(), "binary_test_user");
        assert_eq!(loaded.values[2], 3.25); // Exact match - no precision loss in binary
        assert_eq!(loaded.values[5], 0.001); // Small values preserved exactly

        // Clean up
        fs::remove_file(test_file).unwrap();
    }

    /// Test 3: Complex nested struct binary file roundtrip
    ///
    /// Verifies that a complex nested struct with optional values and deep nesting
    /// can be saved to and loaded from a binary file correctly.
    #[test]
    fn test_complex_nested_struct_binary_file_roundtrip() {
        // Create nested simple structs
        let nested_structs = vec![
            SimpleStruct {
                id: 1,
                name: "binary_nested_1".to_string(),
                active: true,
                score: 85.5,
            },
            SimpleStruct {
                id: 2,
                name: "binary_nested_2".to_string(),
                active: false,
                score: 92.3,
            },
            SimpleStruct {
                id: 3,
                name: "binary_nested_3".to_string(),
                active: true,
                score: 77.8,
            },
        ];

        // Create nested maps
        let mut nested_maps = HashMap::new();
        nested_maps.insert(
            "binary_category_a".to_string(),
            "binary_value_a".to_string(),
        );
        nested_maps.insert(
            "binary_category_b".to_string(),
            "binary_value_b".to_string(),
        );
        nested_maps.insert(
            "binary_category_c".to_string(),
            "binary_value_c".to_string(),
        );

        // Create medium struct for nesting
        let mut medium_metadata = HashMap::new();
        medium_metadata.insert("subsystem".to_string(), "binary_test_subsystem".to_string());
        medium_metadata.insert("level".to_string(), "deep_binary".to_string());

        let mut medium_config = HashMap::new();
        medium_config.insert("nested_binary_config".to_string(), "enabled".to_string());

        let medium_data = MediumStruct {
            metadata: medium_metadata,
            values: vec![10.1, 20.2, 30.3, 40.4],
            tags: vec!["binary_nested".to_string(), "binary_complex".to_string()],
            config: medium_config,
        };

        // Create original complex struct
        let original = ComplexStruct {
            simple_data: SimpleStruct {
                id: 100,
                name: "main_binary_simple".to_string(),
                active: true,
                score: 95.7,
            },
            medium_data,
            nested_structs,
            nested_maps,
            matrix_data: vec![
                1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11, 12.12,
            ],
            matrix_rows: 3,
            matrix_cols: 4,
            optional_data: Some("complex_binary_optional_value".to_string()),
            byte_data: vec![0xAB, 0xCD, 0xEF, 0x12, 0x34, 0x56, 0x78, 0x9A],
        };

        let test_file = "test_complex_nested_struct.bin";

        // Save to file
        original.save_binary(test_file).unwrap();

        // Verify file exists and has content
        assert!(Path::new(test_file).exists());
        let file_metadata = fs::metadata(test_file).unwrap();
        assert!(file_metadata.len() > 100); // Should have substantial binary data

        // Binary files should be more compact than JSON for complex structures
        assert!(file_metadata.len() < 2000);

        // Load from file
        let loaded = ComplexStruct::load_binary(test_file).unwrap();

        // Verify data integrity
        assert_eq!(original, loaded);

        // Verify specific nested fields (binary preserves exact values)
        assert_eq!(loaded.simple_data.name, "main_binary_simple");
        assert_eq!(loaded.nested_structs.len(), 3);
        assert_eq!(loaded.nested_structs[0].name, "binary_nested_1");
        assert_eq!(loaded.nested_structs[1].name, "binary_nested_2");
        assert_eq!(loaded.nested_structs[2].name, "binary_nested_3");
        assert_eq!(loaded.medium_data.tags.len(), 2);
        assert_eq!(loaded.medium_data.tags[0], "binary_nested");
        assert_eq!(
            loaded.nested_maps.get("binary_category_a").unwrap(),
            "binary_value_a"
        );
        assert_eq!(loaded.matrix_data.len(), 12);
        assert_eq!(loaded.matrix_data[2], 3.3); // Exact match - no precision loss
        assert_eq!(loaded.matrix_data[9], 10.10); // Verify additional precision is preserved
        assert_eq!(
            loaded.optional_data.as_ref().unwrap(),
            "complex_binary_optional_value"
        );
        assert_eq!(
            loaded.byte_data,
            vec![0xAB, 0xCD, 0xEF, 0x12, 0x34, 0x56, 0x78, 0x9A]
        );

        // Verify matrix dimensions
        assert_eq!(loaded.matrix_rows, 3);
        assert_eq!(loaded.matrix_cols, 4);

        // Clean up
        fs::remove_file(test_file).unwrap();
    }
}
