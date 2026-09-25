# Android prototype workflow

On 2026-09-25, the local research interface at revision `6f904f49fb1a87a92e08583aef68c5b8c060803c` passed a bounded Android emulator check. This tests the unvalidated dot preview, not the eventual facial-recognition protection method.

The Google Play emulator ran Android 15 / API 35, x86-64, with Chrome 124.0.6367.219, a 1080×2400 display and a 412-pixel CSS viewport. Chrome's reduced user agent said Android 10; the OS version was checked separately through Android. The existing emulator image was launched headlessly with `-read-only -no-snapshot`. Its saved configuration remained unchanged, and the temporary instance and localhost forwarding were stopped afterward.

[`android_workflow.cjs`](android_workflow.cjs) uses the installed [Playwright Android API](https://playwright.dev/docs/api/class-android). A synthetic 640×480 JPEG carries EXIF orientation 6 and a recognizable metadata comment. The check selects it through the file input, confirms a region, processes the preview, and clicks Download JPEG. It verified:

- A 480×640 JPEG was saved by Android's download manager, with neither the original EXIF nor the identifying comment.
- The page fit the viewport and emitted no page errors, external HTTP requests, or requests with a body during the workflow.
- The preview took 2.41 seconds on this small synthetic image. That is not an optimizer, phone-hardware, or final-method benchmark.

The root agent also inspected settled native screenshots of the entry, selection, processed image and download controls. Early full-page captures from this Android/Chrome combination repeated viewport content; they were not used as visual evidence. Android saved the first download even though Playwright's desktop-style download event did not arrive. The final check reads the actual Android file and refuses to run if a file with that name already exists, preventing stale-file success.

For reproduction, start a task-owned emulator and the existing research-mode Vite server, forward device port 5173 to host port 5173 with targeted `adb reverse`, and configure Chrome's documented command-line support for Android automation. Run:

```text
node research/browser/android_workflow.cjs EMULATOR_SERIAL PRIVATE_OUTPUT_DIRECTORY
```

Keep the download location empty for this test; do not delete an unrelated user's file. This run used temporary Android debug-app configuration, restored it, and stopped the read-only emulator. See [Chromium's flag guidance](https://www.chromium.org/developers/how-tos/run-chromium-with-flags/) and the [emulator command-line documentation](https://developer.android.com/studio/run/emulator-commandline).

Executed harness SHA-256: `c653a1f82ad55932ad209d0e882578e13f4e2af45064ab0745a5aa326fa988d7`. Private final report: `70f118014607f50dd4855f85796f05ad6ae0c4d94ce550d723113fdc3461b776`. Downloaded 22,841-byte JPEG: `13a7eb98341b979d95ce4687fd3d087d6d48470b2eaa4afb3cd4cd79a73841ee`. The private run is `FCKFACE-data/runs/android-workflow-v1/attempt4`.

This does not verify the native OS photo picker, a physical Android device, iPhone Safari, model inference, the 90-second target, or public processing. Jordan accepted emulator evidence for Android; the final candidate still needs its own complete workflow check and the offered iPhone test.
