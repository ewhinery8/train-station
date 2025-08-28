//! JSON string escaping and unescaping utilities
//!
//! This module provides functions for escaping and unescaping strings according to the JSON specification.
//! It handles control characters, quotes, backslashes, and Unicode escape sequences.

use crate::serialization::core::{SerializationError, SerializationResult};

/// Escape a string for JSON output
///
/// This function escapes special characters in a string according to the JSON specification.
/// It handles control characters, quotes, backslashes, and other special characters.
///
/// # Arguments
///
/// * `s` - The string to escape
///
/// # Returns
///
/// The escaped string
pub fn escape_string(s: &str) -> String {
    let mut result = String::new();
    for ch in s.chars() {
        match ch {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\x08' => result.push_str("\\b"),
            '\x0c' => result.push_str("\\f"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            ch if ch.is_control() => {
                result.push_str(&format!("\\u{:04x}", ch as u32));
            }
            _ => result.push(ch),
        }
    }
    result
}

/// Unescape a JSON string
///
/// This function unescapes special characters in a JSON string according to the JSON specification.
/// It handles escape sequences like `\n`, `\t`, `\u1234`, etc.
///
/// # Arguments
///
/// * `s` - The string to unescape
///
/// # Returns
///
/// The unescaped string on success, or `SerializationError` on failure
#[allow(unused)]
pub fn unescape_string(s: &str) -> SerializationResult<String> {
    let mut result = String::new();
    let mut chars = s.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\\' {
            if let Some(escaped) = chars.next() {
                match escaped {
                    '"' => result.push('"'),
                    '\\' => result.push('\\'),
                    '/' => result.push('/'),
                    'b' => result.push('\x08'),
                    'f' => result.push('\x0c'),
                    'n' => result.push('\n'),
                    'r' => result.push('\r'),
                    't' => result.push('\t'),
                    'u' => {
                        // Unicode escape sequence
                        let mut code_point = 0u32;
                        for _ in 0..4 {
                            if let Some(hex_digit) = chars.next() {
                                let digit_value = match hex_digit {
                                    '0'..='9' => hex_digit as u32 - '0' as u32,
                                    'a'..='f' => hex_digit as u32 - 'a' as u32 + 10,
                                    'A'..='F' => hex_digit as u32 - 'A' as u32 + 10,
                                    _ => {
                                        return Err(SerializationError::JsonFormat {
                                            message: format!(
                                                "Invalid hex digit in Unicode escape: {}",
                                                hex_digit
                                            ),
                                            line: None,
                                            column: None,
                                        });
                                    }
                                };
                                code_point = code_point * 16 + digit_value;
                            } else {
                                return Err(SerializationError::JsonFormat {
                                    message: "Incomplete Unicode escape sequence".to_string(),
                                    line: None,
                                    column: None,
                                });
                            }
                        }

                        if let Some(ch) = char::from_u32(code_point) {
                            result.push(ch);
                        } else {
                            return Err(SerializationError::JsonFormat {
                                message: format!("Invalid Unicode code point: {}", code_point),
                                line: None,
                                column: None,
                            });
                        }
                    }
                    _ => {
                        return Err(SerializationError::JsonFormat {
                            message: format!("Invalid escape sequence: \\{}", escaped),
                            line: None,
                            column: None,
                        });
                    }
                }
            } else {
                return Err(SerializationError::JsonFormat {
                    message: "Unterminated escape sequence".to_string(),
                    line: None,
                    column: None,
                });
            }
        } else {
            result.push(ch);
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_string() {
        assert_eq!(escape_string("Hello\nWorld"), "Hello\\nWorld");
        assert_eq!(escape_string("Quote: \"Hello\""), "Quote: \\\"Hello\\\"");
        assert_eq!(escape_string("Backslash: \\"), "Backslash: \\\\");
        assert_eq!(escape_string("Tab\there"), "Tab\\there");
        assert_eq!(escape_string("Bell\x07"), "Bell\\u0007");
    }

    #[test]
    fn test_unescape_string() {
        assert_eq!(unescape_string("Hello\\nWorld").unwrap(), "Hello\nWorld");
        assert_eq!(
            unescape_string("Quote: \\\"Hello\\\"").unwrap(),
            "Quote: \"Hello\""
        );
        assert_eq!(unescape_string("Backslash: \\\\").unwrap(), "Backslash: \\");
        assert_eq!(unescape_string("Tab\\there").unwrap(), "Tab\there");
        assert_eq!(unescape_string("Unicode: \\u0041").unwrap(), "Unicode: A");
    }

    #[test]
    fn test_roundtrip_escaping() {
        let original = "Hello\nWorld\twith \"quotes\" and \\backslashes\\";
        let escaped = escape_string(original);
        let unescaped = unescape_string(&escaped).unwrap();
        assert_eq!(unescaped, original);
    }

    #[test]
    fn test_unescape_errors() {
        // Unterminated escape sequence
        assert!(unescape_string("Hello\\").is_err());

        // Invalid escape sequence
        assert!(unescape_string("Hello\\x").is_err());

        // Incomplete Unicode escape
        assert!(unescape_string("Hello\\u123").is_err());

        // Invalid hex digit in Unicode escape
        assert!(unescape_string("Hello\\u123g").is_err());

        // Invalid Unicode code point (0x1100 is valid, but 0x110000 would be invalid if parsed as 6 digits)
        // However, JSON spec only allows 4 hex digits, so \u110000 is parsed as \u1100 followed by "00"
        let result = unescape_string("Hello\\u110000");
        // This should succeed because \u1100 is a valid Unicode character and "00" are literal characters
        assert!(result.is_ok());

        // Test surrogate pairs - these are invalid when used in isolation
        assert!(unescape_string("Hello\\uD800").is_err()); // High surrogate - invalid when used in isolation
        assert!(unescape_string("Hello\\uDFFF").is_err()); // Low surrogate - invalid when used in isolation

        // Test valid code points
        assert!(unescape_string("Hello\\u0041").is_ok()); // 'A' - valid
        assert!(unescape_string("Hello\\u0042").is_ok()); // 'B' - valid
    }

    #[test]
    fn test_escape_comprehensive() {
        // Test all control characters
        let control_chars = "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F";
        let escaped = escape_string(control_chars);
        assert!(escaped.contains("\\u0000"));
        assert!(escaped.contains("\\u0001"));
        assert!(escaped.contains("\\t")); // \x09
        assert!(escaped.contains("\\n")); // \x0A
        assert!(escaped.contains("\\u000b")); // \x0B
        assert!(escaped.contains("\\f")); // \x0C
        assert!(escaped.contains("\\r")); // \x0D

        // Test mixed content
        let mixed = "Hello\nWorld\twith \"quotes\" and \\backslashes\\ and unicode: 🚀";
        let escaped = escape_string(mixed);
        assert!(escaped.contains("\\n"));
        assert!(escaped.contains("\\t"));
        assert!(escaped.contains("\\\""));
        assert!(escaped.contains("\\\\"));
        assert!(escaped.contains("🚀")); // Unicode should be preserved

        // Test empty string
        assert_eq!(escape_string(""), "");
    }

    #[test]
    fn test_unescape_comprehensive() {
        // Test all escape sequences
        let escaped = "\\\"\\\\\\/\\b\\f\\n\\r\\t\\u0041\\u0042";
        let unescaped = unescape_string(escaped).unwrap();
        assert_eq!(unescaped, "\"\\/\x08\x0c\n\r\tAB");

        // Test mixed content with unicode
        let escaped = "Hello\\nWorld\\twith \\\"quotes\\\" and \\\\backslashes\\\\ and unicode: \\u0048\\u0065\\u006c\\u006c\\u006f";
        let unescaped = unescape_string(escaped).unwrap();
        assert_eq!(
            unescaped,
            "Hello\nWorld\twith \"quotes\" and \\backslashes\\ and unicode: Hello"
        );

        // Test empty string
        assert_eq!(unescape_string("").unwrap(), "");
    }

    #[test]
    fn test_escape_roundtrip_comprehensive() {
        // Test various strings with special characters
        let test_strings = vec![
            "Simple string",
            "String with \"quotes\"",
            "String with \\backslashes\\",
            "String with\nnewlines",
            "String with\ttabs",
            "String with\r\ncarriage returns",
            "String with unicode: 🚀🌟✨",
            "Mixed: \"quotes\", \\backslashes\\, \nnewlines, \ttabs, and 🚀unicode🚀",
            "", // Empty string
            "Just spaces   and\ttabs",
        ];

        for original in test_strings {
            let escaped = escape_string(original);
            let unescaped = unescape_string(&escaped).unwrap();
            assert_eq!(unescaped, original, "Roundtrip failed for: {:?}", original);
        }
    }

    #[test]
    fn test_escape_edge_cases() {
        // Test strings with only special characters
        assert_eq!(escape_string("\""), "\\\"");
        assert_eq!(escape_string("\\"), "\\\\");
        assert_eq!(escape_string("\n"), "\\n");
        assert_eq!(escape_string("\t"), "\\t");
        assert_eq!(escape_string("\r"), "\\r");
        assert_eq!(escape_string("\x08"), "\\b");
        assert_eq!(escape_string("\x0c"), "\\f");

        // Test strings with control characters
        assert_eq!(escape_string("\x00"), "\\u0000");
        assert_eq!(escape_string("\x01"), "\\u0001");
        assert_eq!(escape_string("\x1f"), "\\u001f");

        // Test unicode characters
        assert_eq!(escape_string("🚀"), "🚀"); // Should not be escaped
        assert_eq!(escape_string("🌟"), "🌟"); // Should not be escaped
        assert_eq!(escape_string("✨"), "✨"); // Should not be escaped
    }
}
