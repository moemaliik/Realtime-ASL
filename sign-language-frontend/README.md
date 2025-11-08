# Sign Language Recognition Frontend

A modern Svelte.js frontend for the Sign Language Recognition System, built with Tauri for desktop applications.

## Features

- Real-time sign language recognition
- Modern, responsive UI
- Desktop application support via Tauri
- Camera integration
- Hand skeleton visualization
- Word suggestions
- Text-to-speech functionality

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Tauri Commands

This frontend integrates with Tauri for desktop application functionality:

- `npm run tauri dev` - Run in development mode
- `npm run tauri build` - Build desktop application

## Technologies

- **SvelteKit** - Frontend framework
- **Tauri** - Desktop application framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Rust** - Backend (Tauri)

## Project Structure

```
src/
├── routes/
│   └── +page.svelte    # Main application page
├── app.html            # HTML template
├── app.d.ts           # TypeScript definitions
└── hooks.server.ts    # Server hooks

src-tauri/
├── src/
│   └── main.rs        # Rust backend
├── Cargo.toml         # Rust dependencies
└── tauri.conf.json    # Tauri configuration
```

## Usage

1. Start the development server: `npm run dev`
2. Open your browser to `http://localhost:5173`
3. Allow camera access when prompted
4. Make sign language gestures in front of the camera
5. Use the "next" gesture to add characters to your sentence
6. Use word suggestions to correct spelling
7. Click "Speak" to hear the text-to-speech output


