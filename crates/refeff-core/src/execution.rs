//! Cooperative controls shared by engine stages and numerical workers.
use std::cell::RefCell;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

/// A cloneable cancellation handle. Cancellation is permanent for this token.
#[derive(Clone, Default, Debug)]
pub struct CancellationToken(Arc<AtomicBool>);
impl CancellationToken {
    /// Request cancellation at the next cooperative checkpoint.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}
/// Why a calculation stopped cooperatively.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum Interrupted {
    /// Cancellation was requested.
    #[error("calculation cancelled")]
    Cancelled,
    /// Its deadline expired.
    #[error("calculation deadline exceeded")]
    Deadline,
}
/// Settings inherited by the calculation's worker pool.
#[derive(Clone, Default)]
pub struct Control {
    /// Caller-owned cancellation handle.
    pub cancellation: CancellationToken,
    /// Optional monotonic deadline.
    pub deadline: Option<Instant>,
}
impl Control {
    /// Check cancellation and deadline without changing either.
    pub fn check(&self) -> Result<(), Interrupted> {
        if self.cancellation.is_cancelled() {
            return Err(Interrupted::Cancelled);
        }
        if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(Interrupted::Deadline);
        }
        Ok(())
    }
}
thread_local! { static CONTROL: RefCell<Control> = RefCell::new(Control::default()); }
/// Install controls on a newly created worker owned by this calculation.
#[doc(hidden)]
pub fn install_worker(control: Control) {
    CONTROL.with(|current| *current.borrow_mut() = control);
}
/// Run with controls on the current thread, restoring the previous scope afterward.
#[doc(hidden)]
pub fn with_control<T>(control: Control, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<Control>);
    impl Drop for Restore {
        fn drop(&mut self) {
            if let Some(previous) = self.0.take() {
                CONTROL.with(|current| *current.borrow_mut() = previous);
            }
        }
    }
    let _restore = Restore(Some(CONTROL.with(|current| current.replace(control))));
    run()
}
/// Check the controls installed on this numerical worker.
pub fn checkpoint() -> Result<(), Interrupted> {
    CONTROL.with(|current| current.borrow().check())
}
