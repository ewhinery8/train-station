//! Nested Structures Serialization Example
//!
//! This example demonstrates complex serialization patterns with nested structures:
//! - Hierarchical data structures with multiple nesting levels
//! - Structs containing other serializable structs
//! - Collections of complex objects
//! - Advanced field relationships and dependencies
//! - Performance considerations for deep nesting
//!
//! # Learning Objectives
//!
//! - Understand nested struct serialization patterns
//! - Learn to handle complex object hierarchies
//! - Master collection serialization with custom types
//! - Explore performance implications of deep nesting
//! - Implement validation for complex data relationships
//!
//! # Prerequisites
//!
//! - Understanding of basic struct serialization (see basic_structs.rs)
//! - Knowledge of Rust collections (Vec, HashMap)
//! - Familiarity with nested data structures
//!
//! # Usage
//!
//! ```bash
//! cargo run --example nested_structures
//! ```

use std::{collections::HashMap, fs};
use train_station::serialization::{
    FieldValue, FromFieldValue, SerializationError, SerializationResult, StructDeserializer,
    StructSerializable, StructSerializer, ToFieldValue,
};

/// Contact information struct
#[derive(Debug, Clone, PartialEq)]
pub struct ContactInfo {
    pub email: String,
    pub phone: Option<String>,
    pub address_city: String,
    pub address_state: String,
    pub social_media: HashMap<String, String>,
}

impl StructSerializable for ContactInfo {
    fn to_serializer(&self) -> StructSerializer {
        StructSerializer::new()
            .field("email", &self.email)
            .field("phone", &self.phone)
            .field("address_city", &self.address_city)
            .field("address_state", &self.address_state)
            .field("social_media", &self.social_media)
    }

    fn from_deserializer(deserializer: &mut StructDeserializer) -> SerializationResult<Self> {
        let email = deserializer.field("email")?;
        let phone = deserializer.field("phone")?;
        let address_city = deserializer.field("address_city")?;
        let address_state = deserializer.field("address_state")?;
        let social_media = deserializer.field("social_media")?;

        Ok(ContactInfo {
            email,
            phone,
            address_city,
            address_state,
            social_media,
        })
    }
}

impl ToFieldValue for ContactInfo {
    fn to_field_value(&self) -> FieldValue {
        match self.to_json() {
            Ok(json_str) => FieldValue::from_json_object(json_str),
            Err(_) => FieldValue::from_string("serialization_error".to_string()),
        }
    }
}

impl FromFieldValue for ContactInfo {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        // Try JSON object first
        if let Ok(json_data) = value.as_json_object() {
            return Self::from_json(json_data).map_err(|e| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Failed to deserialize ContactInfo from JSON: {}", e),
            });
        }

        // Try binary object
        if let Ok(binary_data) = value.as_binary_object() {
            return Self::from_binary(binary_data).map_err(|e| {
                SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: format!("Failed to deserialize ContactInfo from binary: {}", e),
                }
            });
        }

        Err(SerializationError::ValidationFailed {
            field: field_name.to_string(),
            message: format!(
                "Expected JsonObject or BinaryObject for ContactInfo, found {}",
                value.type_name()
            ),
        })
    }
}

/// Address struct
#[derive(Debug, Clone, PartialEq)]
pub struct Address {
    pub street: String,
    pub city: String,
    pub state: String,
    pub postal_code: String,
    pub country: String,
}

impl StructSerializable for Address {
    fn to_serializer(&self) -> StructSerializer {
        StructSerializer::new()
            .field("street", &self.street)
            .field("city", &self.city)
            .field("state", &self.state)
            .field("postal_code", &self.postal_code)
            .field("country", &self.country)
    }

    fn from_deserializer(deserializer: &mut StructDeserializer) -> SerializationResult<Self> {
        let street = deserializer.field("street")?;
        let city = deserializer.field("city")?;
        let state = deserializer.field("state")?;
        let postal_code = deserializer.field("postal_code")?;
        let country = deserializer.field("country")?;

        Ok(Address {
            street,
            city,
            state,
            postal_code,
            country,
        })
    }
}

impl ToFieldValue for Address {
    fn to_field_value(&self) -> FieldValue {
        match self.to_json() {
            Ok(json_str) => FieldValue::from_json_object(json_str),
            Err(_) => FieldValue::from_string("serialization_error".to_string()),
        }
    }
}

impl FromFieldValue for Address {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        // Try JSON object first
        if let Ok(json_data) = value.as_json_object() {
            return Self::from_json(json_data).map_err(|e| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Failed to deserialize Address from JSON: {}", e),
            });
        }

        // Try binary object
        if let Ok(binary_data) = value.as_binary_object() {
            return Self::from_binary(binary_data).map_err(|e| {
                SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: format!("Failed to deserialize Address from binary: {}", e),
                }
            });
        }

        Err(SerializationError::ValidationFailed {
            field: field_name.to_string(),
            message: format!(
                "Expected JsonObject or BinaryObject for Address, found {}",
                value.type_name()
            ),
        })
    }
}

/// Project information struct
#[derive(Debug, Clone, PartialEq)]
pub struct Project {
    pub name: String,
    pub description: String,
    pub status: ProjectStatus,
    pub budget: f64,
    pub team_members: Vec<String>,
    pub milestones: Vec<Milestone>,
    pub metadata: HashMap<String, String>,
}

impl StructSerializable for Project {
    fn to_serializer(&self) -> StructSerializer {
        StructSerializer::new()
            .field("name", &self.name)
            .field("description", &self.description)
            .field("status", &self.status)
            .field("budget", &self.budget)
            .field("team_members", &self.team_members)
            .field("milestones", &self.milestones)
            .field("metadata", &self.metadata)
    }

    fn from_deserializer(deserializer: &mut StructDeserializer) -> SerializationResult<Self> {
        let name = deserializer.field("name")?;
        let description = deserializer.field("description")?;
        let status = deserializer.field("status")?;
        let budget = deserializer.field("budget")?;
        let team_members = deserializer.field("team_members")?;
        let milestones = deserializer.field("milestones")?;
        let metadata = deserializer.field("metadata")?;

        Ok(Project {
            name,
            description,
            status,
            budget,
            team_members,
            milestones,
            metadata,
        })
    }
}

impl ToFieldValue for Project {
    fn to_field_value(&self) -> FieldValue {
        match self.to_json() {
            Ok(json_str) => FieldValue::from_json_object(json_str),
            Err(_) => FieldValue::from_string("serialization_error".to_string()),
        }
    }
}

impl FromFieldValue for Project {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        // Try JSON object first
        if let Ok(json_data) = value.as_json_object() {
            return Self::from_json(json_data).map_err(|e| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Failed to deserialize Project from JSON: {}", e),
            });
        }

        // Try binary object
        if let Ok(binary_data) = value.as_binary_object() {
            return Self::from_binary(binary_data).map_err(|e| {
                SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: format!("Failed to deserialize Project from binary: {}", e),
                }
            });
        }

        Err(SerializationError::ValidationFailed {
            field: field_name.to_string(),
            message: format!(
                "Expected JsonObject or BinaryObject for Project, found {}",
                value.type_name()
            ),
        })
    }
}

/// Project status enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectStatus {
    Planning,
    InProgress,
    OnHold,
    Completed,
    Cancelled,
}

impl ToFieldValue for ProjectStatus {
    fn to_field_value(&self) -> FieldValue {
        let status_str = match self {
            ProjectStatus::Planning => "planning",
            ProjectStatus::InProgress => "in_progress",
            ProjectStatus::OnHold => "on_hold",
            ProjectStatus::Completed => "completed",
            ProjectStatus::Cancelled => "cancelled",
        };
        FieldValue::from_string(status_str.to_string())
    }
}

impl FromFieldValue for ProjectStatus {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        match value {
            FieldValue::String(s) => match s.as_str() {
                "planning" => Ok(ProjectStatus::Planning),
                "in_progress" => Ok(ProjectStatus::InProgress),
                "on_hold" => Ok(ProjectStatus::OnHold),
                "completed" => Ok(ProjectStatus::Completed),
                "cancelled" => Ok(ProjectStatus::Cancelled),
                _ => Err(SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: format!("Unknown project status: {}", s),
                }),
            },
            _ => Err(SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!(
                    "Expected String for ProjectStatus, found {}",
                    value.type_name()
                ),
            }),
        }
    }
}

/// Project milestone struct
#[derive(Debug, Clone, PartialEq)]
pub struct Milestone {
    pub name: String,
    pub description: String,
    pub due_date: String, // Simplified as string for this example
    pub is_completed: bool,
    pub progress_percentage: f32,
    pub dependencies: Vec<String>,
}

impl StructSerializable for Milestone {
    fn to_serializer(&self) -> StructSerializer {
        StructSerializer::new()
            .field("name", &self.name)
            .field("description", &self.description)
            .field("due_date", &self.due_date)
            .field("is_completed", &self.is_completed)
            .field("progress_percentage", &self.progress_percentage)
            .field("dependencies", &self.dependencies)
    }

    fn from_deserializer(deserializer: &mut StructDeserializer) -> SerializationResult<Self> {
        let name = deserializer.field("name")?;
        let description = deserializer.field("description")?;
        let due_date = deserializer.field("due_date")?;
        let is_completed = deserializer.field("is_completed")?;
        let progress_percentage = deserializer.field("progress_percentage")?;
        let dependencies = deserializer.field("dependencies")?;

        Ok(Milestone {
            name,
            description,
            due_date,
            is_completed,
            progress_percentage,
            dependencies,
        })
    }
}

impl ToFieldValue for Milestone {
    fn to_field_value(&self) -> FieldValue {
        match self.to_json() {
            Ok(json_str) => FieldValue::from_json_object(json_str),
            Err(_) => FieldValue::from_string("serialization_error".to_string()),
        }
    }
}

impl FromFieldValue for Milestone {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        // Try JSON object first
        if let Ok(json_data) = value.as_json_object() {
            return Self::from_json(json_data).map_err(|e| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Failed to deserialize Milestone from JSON: {}", e),
            });
        }

        // Try binary object
        if let Ok(binary_data) = value.as_binary_object() {
            return Self::from_binary(binary_data).map_err(|e| {
                SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: format!("Failed to deserialize Milestone from binary: {}", e),
                }
            });
        }

        Err(SerializationError::ValidationFailed {
            field: field_name.to_string(),
            message: format!(
                "Expected JsonObject or BinaryObject for Milestone, found {}",
                value.type_name()
            ),
        })
    }
}

/// Company struct with basic collections and nesting
#[derive(Debug, Clone, PartialEq)]
pub struct Company {
    pub name: String,
    pub founded_year: i32,
    pub headquarters_city: String,
    pub headquarters_state: String,
    pub employee_count: usize,
    pub department_names: Vec<String>,
    pub active_project_names: Vec<String>,
    pub company_metadata: HashMap<String, String>,
}

impl StructSerializable for Company {
    fn to_serializer(&self) -> StructSerializer {
        StructSerializer::new()
            .field("name", &self.name)
            .field("founded_year", &self.founded_year)
            .field("headquarters_city", &self.headquarters_city)
            .field("headquarters_state", &self.headquarters_state)
            .field("employee_count", &self.employee_count)
            .field("department_names", &self.department_names)
            .field("active_project_names", &self.active_project_names)
            .field("company_metadata", &self.company_metadata)
    }

    fn from_deserializer(deserializer: &mut StructDeserializer) -> SerializationResult<Self> {
        let name = deserializer.field("name")?;
        let founded_year = deserializer.field("founded_year")?;
        let headquarters_city = deserializer.field("headquarters_city")?;
        let headquarters_state = deserializer.field("headquarters_state")?;
        let employee_count = deserializer.field("employee_count")?;
        let department_names = deserializer.field("department_names")?;
        let active_project_names = deserializer.field("active_project_names")?;
        let company_metadata = deserializer.field("company_metadata")?;

        Ok(Company {
            name,
            founded_year,
            headquarters_city,
            headquarters_state,
            employee_count,
            department_names,
            active_project_names,
            company_metadata,
        })
    }
}

impl ToFieldValue for Company {
    fn to_field_value(&self) -> FieldValue {
        match self.to_json() {
            Ok(json_str) => FieldValue::from_json_object(json_str),
            Err(_) => FieldValue::from_string("serialization_error".to_string()),
        }
    }
}

impl FromFieldValue for Company {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        // Try JSON object first
        if let Ok(json_data) = value.as_json_object() {
            return Self::from_json(json_data).map_err(|e| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Failed to deserialize Company from JSON: {}", e),
            });
        }

        // Try binary object
        if let Ok(binary_data) = value.as_binary_object() {
            return Self::from_binary(binary_data).map_err(|e| {
                SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: format!("Failed to deserialize Company from binary: {}", e),
                }
            });
        }

        Err(SerializationError::ValidationFailed {
            field: field_name.to_string(),
            message: format!(
                "Expected JsonObject or BinaryObject for Company, found {}",
                value.type_name()
            ),
        })
    }
}

/// Department struct
#[derive(Debug, Clone, PartialEq)]
pub struct Department {
    pub name: String,
    pub manager: String,
    pub employee_count: u32,
    pub budget: f64,
    pub office_locations: Vec<Address>,
}

impl StructSerializable for Department {
    fn to_serializer(&self) -> StructSerializer {
        StructSerializer::new()
            .field("name", &self.name)
            .field("manager", &self.manager)
            .field("employee_count", &self.employee_count)
            .field("budget", &self.budget)
            .field("office_locations", &self.office_locations)
    }

    fn from_deserializer(deserializer: &mut StructDeserializer) -> SerializationResult<Self> {
        let name = deserializer.field("name")?;
        let manager = deserializer.field("manager")?;
        let employee_count = deserializer.field("employee_count")?;
        let budget = deserializer.field("budget")?;
        let office_locations = deserializer.field("office_locations")?;

        Ok(Department {
            name,
            manager,
            employee_count,
            budget,
            office_locations,
        })
    }
}

impl ToFieldValue for Department {
    fn to_field_value(&self) -> FieldValue {
        match self.to_json() {
            Ok(json_str) => FieldValue::from_json_object(json_str),
            Err(_) => FieldValue::from_string("serialization_error".to_string()),
        }
    }
}

impl FromFieldValue for Department {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        // Try JSON object first
        if let Ok(json_data) = value.as_json_object() {
            return Self::from_json(json_data).map_err(|e| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Failed to deserialize Department from JSON: {}", e),
            });
        }

        // Try binary object
        if let Ok(binary_data) = value.as_binary_object() {
            return Self::from_binary(binary_data).map_err(|e| {
                SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: format!("Failed to deserialize Department from binary: {}", e),
                }
            });
        }

        Err(SerializationError::ValidationFailed {
            field: field_name.to_string(),
            message: format!(
                "Expected JsonObject or BinaryObject for Department, found {}",
                value.type_name()
            ),
        })
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Nested Structures Serialization Example ===\n");

    demonstrate_nested_struct_creation()?;
    demonstrate_deep_serialization()?;
    demonstrate_collection_nesting()?;
    demonstrate_partial_loading()?;
    demonstrate_performance_analysis()?;
    cleanup_temp_files()?;

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

/// Demonstrate creating complex nested structures
fn demonstrate_nested_struct_creation() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Nested Structure Creation ---");

    // Create nested address and contact info
    let headquarters = Address {
        street: "123 Innovation Drive".to_string(),
        city: "Tech City".to_string(),
        state: "CA".to_string(),
        postal_code: "94000".to_string(),
        country: "USA".to_string(),
    };

    let mut social_media = HashMap::new();
    social_media.insert("twitter".to_string(), "@techcorp".to_string());
    social_media.insert("linkedin".to_string(), "techcorp-inc".to_string());

    let contact_info = ContactInfo {
        email: "info@techcorp.com".to_string(),
        phone: Some("+1-555-0123".to_string()),
        address_city: headquarters.city.clone(),
        address_state: headquarters.state.clone(),
        social_media,
    };

    // Create departments with nested office locations
    let engineering_office = Address {
        street: "456 Developer Lane".to_string(),
        city: "Code City".to_string(),
        state: "CA".to_string(),
        postal_code: "94001".to_string(),
        country: "USA".to_string(),
    };

    let departments = [
        Department {
            name: "Engineering".to_string(),
            manager: "Alice Johnson".to_string(),
            employee_count: 50,
            budget: 2500000.0,
            office_locations: vec![engineering_office, headquarters.clone()],
        },
        Department {
            name: "Marketing".to_string(),
            manager: "Bob Smith".to_string(),
            employee_count: 15,
            budget: 800000.0,
            office_locations: vec![headquarters.clone()],
        },
    ];

    // Create projects with milestones
    let milestones = vec![
        Milestone {
            name: "Requirements Analysis".to_string(),
            description: "Complete system requirements documentation".to_string(),
            due_date: "2024-03-15".to_string(),
            is_completed: true,
            progress_percentage: 100.0,
            dependencies: vec![],
        },
        Milestone {
            name: "Architecture Design".to_string(),
            description: "Define system architecture and components".to_string(),
            due_date: "2024-04-01".to_string(),
            is_completed: false,
            progress_percentage: 75.0,
            dependencies: vec!["Requirements Analysis".to_string()],
        },
    ];

    let mut project_metadata = HashMap::new();
    project_metadata.insert("priority".to_string(), "high".to_string());
    project_metadata.insert("client".to_string(), "internal".to_string());

    let projects = [Project {
        name: "Train Station ML Platform".to_string(),
        description: "Next-generation machine learning infrastructure".to_string(),
        status: ProjectStatus::InProgress,
        budget: 1500000.0,
        team_members: vec![
            "Alice Johnson".to_string(),
            "Charlie Brown".to_string(),
            "Diana Prince".to_string(),
        ],
        milestones: milestones.clone(),
        metadata: project_metadata,
    }];

    // Create the complete company structure
    let mut company_metadata = HashMap::new();
    company_metadata.insert("industry".to_string(), "technology".to_string());
    company_metadata.insert("stock_symbol".to_string(), "TECH".to_string());

    let company = Company {
        name: "TechCorp Inc.".to_string(),
        founded_year: 2015,
        headquarters_city: headquarters.city.clone(),
        headquarters_state: headquarters.state.clone(),
        employee_count: 250,
        department_names: departments.iter().map(|d| d.name.clone()).collect(),
        active_project_names: projects.iter().map(|p| p.name.clone()).collect(),
        company_metadata,
    };

    println!("Created complex company structure:");
    println!("  Company: {}", company.name);
    println!("  Founded: {}", company.founded_year);
    println!(
        "  Headquarters: {}, {}",
        company.headquarters_city, company.headquarters_state
    );
    println!("  Employee Count: {}", company.employee_count);
    println!("  Departments: {}", company.department_names.len());
    println!("  Active Projects: {}", company.active_project_names.len());

    // Save the complete structure
    company.save_json("temp_nested_company.json")?;
    println!("Saved nested structure to: temp_nested_company.json");

    // Verify loading preserves all nested data
    let loaded_company = Company::load_json("temp_nested_company.json")?;
    assert_eq!(company, loaded_company);
    println!("Successfully verified Company roundtrip serialization");

    // Also demonstrate individual component serialization
    let address_json = headquarters.to_json()?;
    let loaded_address = Address::from_json(&address_json)?;
    assert_eq!(headquarters, loaded_address);
    println!("Successfully serialized/deserialized Address component");

    let contact_json = contact_info.to_json()?;
    let loaded_contact = ContactInfo::from_json(&contact_json)?;
    assert_eq!(contact_info, loaded_contact);
    println!("Successfully serialized/deserialized ContactInfo component");
    println!("Nested structure integrity: VERIFIED");

    Ok(())
}

/// Demonstrate deep serialization with complex nesting
fn demonstrate_deep_serialization() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Deep Serialization Analysis ---");

    let deep_milestone = Milestone {
        name: "Deep Milestone".to_string(),
        description: "Testing deep nesting serialization".to_string(),
        due_date: "2024-12-31".to_string(),
        is_completed: false,
        progress_percentage: 50.0,
        dependencies: vec!["Parent Task".to_string(), "Sibling Task".to_string()],
    };

    let deep_project = Project {
        name: "Deep Nesting Test".to_string(),
        description: "Project for testing serialization depth".to_string(),
        status: ProjectStatus::Planning,
        budget: 100000.0,
        team_members: vec!["Developer 1".to_string(), "Developer 2".to_string()],
        milestones: vec![deep_milestone],
        metadata: HashMap::new(),
    };

    // Analyze serialization output
    let json_output = deep_project.to_json()?;
    let binary_output = deep_project.to_binary()?;

    println!("Deep structure serialization analysis:");
    println!("  JSON size: {} bytes", json_output.len());
    println!("  Binary size: {} bytes", binary_output.len());
    println!("  Nesting levels: Address -> Project -> Milestone -> Dependencies");

    // Count nested objects in JSON (rough estimate)
    let object_count = json_output.matches('{').count();
    let array_count = json_output.matches('[').count();
    println!("  JSON objects: {}", object_count);
    println!("  JSON arrays: {}", array_count);

    // Verify deep roundtrip
    let json_parsed = Project::from_json(&json_output)?;
    let binary_parsed = Project::from_binary(&binary_output)?;

    assert_eq!(deep_project, json_parsed);
    assert_eq!(deep_project, binary_parsed);
    println!("Deep serialization roundtrip: VERIFIED");

    Ok(())
}

/// Demonstrate collection nesting patterns
fn demonstrate_collection_nesting() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Collection Nesting Patterns ---");

    // Create multiple departments with varying complexity
    let departments = vec![
        Department {
            name: "Research".to_string(),
            manager: "Dr. Science".to_string(),
            employee_count: 25,
            budget: 1200000.0,
            office_locations: vec![
                Address {
                    street: "1 Research Blvd".to_string(),
                    city: "Innovation Hub".to_string(),
                    state: "MA".to_string(),
                    postal_code: "02101".to_string(),
                    country: "USA".to_string(),
                },
                Address {
                    street: "2 Lab Street".to_string(),
                    city: "Tech Valley".to_string(),
                    state: "NY".to_string(),
                    postal_code: "12180".to_string(),
                    country: "USA".to_string(),
                },
            ],
        },
        Department {
            name: "Quality Assurance".to_string(),
            manager: "Test Master".to_string(),
            employee_count: 12,
            budget: 600000.0,
            office_locations: vec![], // Empty collection
        },
    ];

    println!("Collection nesting analysis:");
    println!("  Departments: {}", departments.len());

    let total_locations: usize = departments.iter().map(|d| d.office_locations.len()).sum();
    println!("  Total office locations: {}", total_locations);

    // Test serialization with mixed empty and populated collections
    // Note: Vec<Department> doesn't implement StructSerializable directly.
    // For this example, we'll serialize each department individually
    let department_json_strings: Result<Vec<String>, _> =
        departments.iter().map(|dept| dept.to_json()).collect();
    let department_json_strings = department_json_strings?;

    // Deserialize each department back
    let parsed_departments: Result<Vec<Department>, _> = department_json_strings
        .iter()
        .map(|json_str| Department::from_json(json_str))
        .collect();
    let parsed_departments = parsed_departments?;

    assert_eq!(departments, parsed_departments);
    println!("Collection nesting serialization: VERIFIED");

    // Analyze collection patterns
    for (i, dept) in departments.iter().enumerate() {
        println!(
            "  Department {}: {} locations",
            i + 1,
            dept.office_locations.len()
        );
    }

    Ok(())
}

/// Demonstrate partial loading and field access
fn demonstrate_partial_loading() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Partial Loading and Field Access ---");

    // Create a simple project for analysis
    let project = Project {
        name: "Sample Project".to_string(),
        description: "For testing partial loading".to_string(),
        status: ProjectStatus::InProgress,
        budget: 50000.0,
        team_members: vec!["Alice".to_string(), "Bob".to_string()],
        milestones: vec![Milestone {
            name: "Phase 1".to_string(),
            description: "Initial phase".to_string(),
            due_date: "2024-06-01".to_string(),
            is_completed: true,
            progress_percentage: 100.0,
            dependencies: vec![],
        }],
        metadata: HashMap::new(),
    };

    // Convert to JSON and analyze structure
    println!("Project JSON structure analysis:");

    // Parse to examine available fields by inspecting JSON structure
    let json_data = project.to_json()?;
    let field_count = json_data.matches(':').count();
    println!("  Estimated fields: {}", field_count);

    // Show top-level structure
    let lines: Vec<&str> = json_data.lines().take(10).collect();
    println!("  JSON structure preview:");
    for line in lines.iter().take(5) {
        if let Some(colon_pos) = line.find(':') {
            let field_name = line[..colon_pos].trim().trim_matches('"').trim();
            if !field_name.is_empty() {
                println!("    - {}", field_name);
            }
        }
    }

    // Demonstrate field type analysis
    println!("\nField type analysis:");
    println!("  name: String");
    println!("  status: Enum -> String");
    println!("  budget: f64 -> Number");
    println!("  team_members: Vec<String> -> Array");
    println!("  milestones: Vec<Milestone> -> Array of Objects");

    Ok(())
}

/// Demonstrate performance analysis for nested structures
fn demonstrate_performance_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Performance Analysis ---");

    // Create structures of varying complexity
    let simple_address = Address {
        street: "123 Main St".to_string(),
        city: "Anytown".to_string(),
        state: "ST".to_string(),
        postal_code: "12345".to_string(),
        country: "USA".to_string(),
    };

    let complex_department = Department {
        name: "Complex Department".to_string(),
        manager: "Manager Name".to_string(),
        employee_count: 100,
        budget: 5000000.0,
        office_locations: vec![simple_address.clone(); 10], // 10 identical addresses
    };

    let complex_project = Project {
        name: "Complex Project".to_string(),
        description: "Large project with many components".to_string(),
        status: ProjectStatus::InProgress,
        budget: 2000000.0,
        team_members: (1..=50).map(|i| format!("Team Member {}", i)).collect(),
        milestones: (1..=20)
            .map(|i| Milestone {
                name: format!("Milestone {}", i),
                description: format!("Description for milestone {}", i),
                due_date: "2024-12-31".to_string(),
                is_completed: i <= 10,
                progress_percentage: if i <= 10 { 100.0 } else { 50.0 },
                dependencies: if i > 1 {
                    vec![format!("Milestone {}", i - 1)]
                } else {
                    vec![]
                },
            })
            .collect(),
        metadata: HashMap::new(),
    };

    // Measure serialization performance
    println!("Performance comparison:");

    // Simple address
    let addr_json = simple_address.to_json()?;
    let addr_binary = simple_address.to_binary()?;
    println!("  Simple Address:");
    println!("    JSON: {} bytes", addr_json.len());
    println!("    Binary: {} bytes", addr_binary.len());

    // Complex department
    let dept_json = complex_department.to_json()?;
    let dept_binary = complex_department.to_binary()?;
    println!("  Complex Department (10 addresses):");
    println!("    JSON: {} bytes", dept_json.len());
    println!("    Binary: {} bytes", dept_binary.len());

    // Complex project
    let proj_json = complex_project.to_json()?;
    let proj_binary = complex_project.to_binary()?;
    println!("  Complex Project (50 members, 20 milestones):");
    println!("    JSON: {} bytes", proj_json.len());
    println!("    Binary: {} bytes", proj_binary.len());

    // Calculate efficiency ratios
    let dept_ratio = dept_json.len() as f64 / dept_binary.len() as f64;
    let proj_ratio = proj_json.len() as f64 / proj_binary.len() as f64;

    println!("\nFormat efficiency (JSON/Binary ratio):");
    println!("  Department: {:.2}x", dept_ratio);
    println!("  Project: {:.2}x", proj_ratio);

    // Verify complex structure roundtrip
    let proj_json_parsed = Project::from_json(&proj_json)?;
    let proj_binary_parsed = Project::from_binary(&proj_binary)?;

    assert_eq!(complex_project, proj_json_parsed);
    assert_eq!(complex_project, proj_binary_parsed);
    println!("Complex structure roundtrip: VERIFIED");

    Ok(())
}

/// Clean up temporary files created during the example
fn cleanup_temp_files() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Cleanup ---");

    let files_to_remove = ["temp_nested_company.json"];

    for file in &files_to_remove {
        if fs::metadata(file).is_ok() {
            fs::remove_file(file)?;
            println!("Removed: {}", file);
        }
    }

    println!("Cleanup completed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_address_serialization() {
        let address = Address {
            street: "123 Test St".to_string(),
            city: "Test City".to_string(),
            state: "TS".to_string(),
            postal_code: "12345".to_string(),
            country: "Test Country".to_string(),
        };

        let json_data = address.to_json().unwrap();
        let parsed_address = Address::from_json(&json_data).unwrap();
        assert_eq!(address, parsed_address);
    }

    #[test]
    fn test_project_status_enum() {
        let statuses = vec![
            ProjectStatus::Planning,
            ProjectStatus::InProgress,
            ProjectStatus::OnHold,
            ProjectStatus::Completed,
            ProjectStatus::Cancelled,
        ];

        for status in statuses {
            let field_value = status.to_field_value();
            let parsed_status = ProjectStatus::from_field_value(field_value, "status").unwrap();
            assert_eq!(status, parsed_status);
        }
    }

    #[test]
    fn test_nested_company_structure() {
        let address = Address {
            street: "Test St".to_string(),
            city: "Test City".to_string(),
            state: "TS".to_string(),
            postal_code: "12345".to_string(),
            country: "Test".to_string(),
        };

        let contact_info = ContactInfo {
            email: "test@test.com".to_string(),
            phone: None,
            address_city: address.city.clone(),
            address_state: address.state.clone(),
            social_media: HashMap::new(),
        };

        let company = Company {
            name: "Test Company".to_string(),
            founded_year: 2020,
            headquarters_city: address.city.clone(),
            headquarters_state: address.state.clone(),
            employee_count: 10,
            department_names: vec![],
            active_project_names: vec![],
            company_metadata: HashMap::new(),
        };

        let json_data = company.to_json().unwrap();
        let parsed_company = Company::from_json(&json_data).unwrap();
        assert_eq!(company, parsed_company);
    }

    #[test]
    fn test_milestone_with_dependencies() {
        let milestone = Milestone {
            name: "Test Milestone".to_string(),
            description: "Test Description".to_string(),
            due_date: "2024-01-01".to_string(),
            is_completed: false,
            progress_percentage: 25.0,
            dependencies: vec!["Dep1".to_string(), "Dep2".to_string()],
        };

        let binary_data = milestone.to_binary().unwrap();
        let parsed_milestone = Milestone::from_binary(&binary_data).unwrap();
        assert_eq!(milestone, parsed_milestone);
    }
}
