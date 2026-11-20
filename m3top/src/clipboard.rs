use std::{
    io::{self, Write},
    process::Stdio,
    time::Duration,
};

use anyhow::Result;
use base64::{Engine, engine::general_purpose::STANDARD};
use tokio::{io::AsyncWriteExt, process::Command, time::timeout};

pub enum CopyMethod {
    Desktop,
    Terminal,
}

pub async fn copy(contents: &str) -> Result<CopyMethod> {
    if std::env::var_os("WAYLAND_DISPLAY").is_some() && try_command("wl-copy", &[], contents).await
    {
        return Ok(CopyMethod::Desktop);
    }
    if std::env::var_os("DISPLAY").is_some() {
        for (program, args) in [
            ("xclip", &["-selection", "clipboard"][..]),
            ("xsel", &["--clipboard", "--input"][..]),
        ] {
            if try_command(program, args, contents).await {
                return Ok(CopyMethod::Desktop);
            }
        }
    }
    if cfg!(target_os = "macos") && try_command("pbcopy", &[], contents).await {
        return Ok(CopyMethod::Desktop);
    }
    let mut output = io::stdout().lock();
    output.write_all(terminal_sequence(contents).as_bytes())?;
    output.flush()?;
    Ok(CopyMethod::Terminal)
}

async fn try_command(program: &str, args: &[&str], contents: &str) -> bool {
    let Ok(mut child) = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
    else {
        return false;
    };
    let attempt = async {
        let mut input = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::other("clipboard input unavailable"))?;
        input.write_all(contents.as_bytes()).await?;
        drop(input);
        Ok::<_, io::Error>(child.wait().await?.success())
    };
    matches!(timeout(Duration::from_secs(2), attempt).await, Ok(Ok(true)))
}

fn terminal_sequence(contents: &str) -> String {
    format!("\x1b]52;c;{}\x07", STANDARD.encode(contents))
}

#[cfg(test)]
mod tests {
    #[test]
    fn terminal_clipboard_encodes_unicode_and_control_characters() {
        assert_eq!(super::terminal_sequence("hello"), "\x1b]52;c;aGVsbG8=\x07");
        let sequence = super::terminal_sequence("é\n\x1b");
        assert_eq!(sequence, "\x1b]52;c;w6kKGw==\x07");
    }
}
