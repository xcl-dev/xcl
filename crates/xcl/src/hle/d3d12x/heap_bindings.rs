// Inline storage for the two shader-visible heap types. Preserve larger input lists.
// Motivation: Vice-Tay 076ad59 and XaviFortes/VIctory-RDNA c43982e.
// Independently implemented for XCL; no external implementation copied.
use std::ops::Deref;

#[derive(Debug, PartialEq)]
pub(super) struct HeapBindings<T> {
    inline: [T; 2],
    len: usize,
    overflow: Vec<T>,
}

impl<T: Default> Default for HeapBindings<T> {
    fn default() -> Self {
        Self { inline: std::array::from_fn(|_| T::default()), len: 0, overflow: Vec::new() }
    }
}

impl<T: Default> HeapBindings<T> {
    pub(super) fn push(&mut self, value: T) {
        if self.len < 2 {
            self.inline[self.len] = value;
        } else {
            if self.len == 2 {
                self.overflow.reserve(3);
                self.overflow.extend(self.inline.iter_mut().map(std::mem::take));
            }
            self.overflow.push(value);
        }
        self.len += 1;
    }
}

impl<T> Deref for HeapBindings<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        if self.len <= 2 { &self.inline[..self.len] } else { &self.overflow }
    }
}

impl<T: Default> FromIterator<T> for HeapBindings<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut out = Self::default();
        for value in iter { out.push(value); }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heap_combinations_stay_inline_and_ordered() {
        for (view, sampler) in [(None, None), (Some(11), None), (None, Some(22)), (Some(11), Some(22))] {
            let heaps: HeapBindings<_> = view.into_iter().chain(sampler).collect();
            assert_eq!(&*heaps, view.into_iter().chain(sampler).collect::<Vec<_>>());
            assert_eq!(heaps.overflow.capacity(), 0);
        }
    }

    #[test]
    fn larger_lists_preserve_order_and_ownership() {
        use std::rc::Rc;
        let owners: Vec<_> = (0..4).map(Rc::new).collect();
        let heaps: HeapBindings<Option<Rc<i32>>> = owners.iter().cloned().map(Some).collect();
        assert_eq!(heaps.iter().map(|h| **h.as_ref().unwrap()).collect::<Vec<_>>(), [0, 1, 2, 3]);
        assert!(owners.iter().all(|h| Rc::strong_count(h) == 2));
        drop(heaps);
        assert!(owners.iter().all(|h| Rc::strong_count(h) == 1));
    }
}
