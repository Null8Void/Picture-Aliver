/**
 * Picture-Aliver Electron - Main Process
 * 
 * Manages the Electron application lifecycle and spawns the backend server.
 */

const { app, BrowserWindow, ipcMain, Menu, shell } = require('electron');
const path = require('path');
const { spawn, exec } = require('child_process');
const log = require('electron-log');

// Configure logging
log.transports.file.level = 'info';
log.transports.console.level = 'debug';
log.info('Picture-Aliver Electron starting...');

// Global references
let mainWindow = null;
let backendProcess = null;
let isQuitting = false;

// Configuration
const BACKEND_PORT = 8000;
const BACKEND_START_DELAY = 3000; // ms to wait for backend

/**
 * Start the FastAPI backend server
 */
function startBackend() {
    log.info('Starting backend server...');
    
    const isDev = !app.isPackaged;
    
    if (isDev) {
        // Development mode - use local Python
        const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
        
        backendProcess = spawn(pythonCmd, [
            '-m', 'uvicorn',
            'src.picture_aliver.api:app',
            '--host', '127.0.0.1',
            '--port', String(BACKEND_PORT),
            '--log-level', 'warning'
        ], {
            cwd: path.join(__dirname, '../..'),
            stdio: ['ignore', 'pipe', 'pipe'],
            detached: false
        });
    } else {
        // Production mode - bundled
        // Note: For production, you'd bundle the Python environment
        const pythonCmd = process.platform === 'win32' ? 'python.exe' : 'python3';
        
        backendProcess = spawn(pythonCmd, [
            '-m', 'uvicorn',
            'src.picture_aliver.api:app',
            '--host', '127.0.0.1',
            '--port', String(BACKEND_PORT),
            '--log-level', 'warning'
        ], {
            cwd: path.join(process.resourcesPath || __dirname, '../..'),
            stdio: ['ignore', 'pipe', 'pipe'],
            detached: false
        });
    }
    
    // Handle backend output
    backendProcess.stdout.on('data', (data) => {
        log.info(`Backend: ${data.toString().trim()}`);
    });
    
    backendProcess.stderr.on('data', (data) => {
        log.warn(`Backend warning: ${data.toString().trim()}`);
    });
    
    backendProcess.on('error', (err) => {
        log.error('Backend process error:', err);
    });
    
    backendProcess.on('exit', (code) => {
        log.info(`Backend exited with code ${code}`);
        if (!isQuitting) {
            log.warn('Backend crashed, attempting restart...');
            setTimeout(startBackend, 2000);
        }
    });
    
    // Wait for backend to start
    setTimeout(() => {
        log.info('Backend should be ready');
    }, BACKEND_START_DELAY);
}

/**
 * Stop the backend server
 */
function stopBackend() {
    if (backendProcess) {
        log.info('Stopping backend...');
        try {
            if (process.platform === 'win32') {
                exec(`taskkill /pid ${backendProcess.pid} /T /F`);
            } else {
                process.kill(-backendProcess.pid);
            }
        } catch (e) {
            log.warn('Error stopping backend:', e);
        }
        backendProcess = null;
    }
}

/**
 * Create the main application window
 */
function createWindow() {
    log.info('Creating main window...');
    
    mainWindow = new BrowserWindow({
        width: 1280,
        height: 800,
        minWidth: 900,
        minHeight: 600,
        title: 'Picture-Aliver',
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js')
        },
        show: false,
        backgroundColor: '#F9FAFB'
    });
    
    // Load the app
    const isDev = !app.isPackaged;
    
    if (isDev) {
        // Development: Load from dist folder or React dev server
        const distPath = path.join(__dirname, '../dist');
        const indexPath = path.join(distPath, 'index.html');
        
        try {
            mainWindow.loadFile(indexPath);
            log.info('Loaded from dist folder');
        } catch (e) {
            // Fallback: load from web
            mainWindow.loadURL('http://localhost:3000');
            log.info('Loaded from localhost:3000');
        }
    } else {
        // Production: Load bundled files
        mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
    }
    
    // Show when ready
    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
        log.info('Window shown');
    });
    
    // Handle window close
    mainWindow.on('close', (event) => {
        if (!isQuitting) {
            event.preventDefault();
            mainWindow.hide();
        }
    });
    
    mainWindow.on('closed', () => {
        mainWindow = null;
    });
    
    // Create application menu
    createMenu();
}

/**
 * Create the application menu
 */
function createMenu() {
    const template = [
        {
            label: 'File',
            submenu: [
                {
                    label: 'Open Image',
                    accelerator: 'CmdOrCtrl+O',
                    click: () => {
                        mainWindow.webContents.send('menu-action', 'open-image');
                    }
                },
                {
                    label: 'Save Video',
                    accelerator: 'CmdOrCtrl+S',
                    click: () => {
                        mainWindow.webContents.send('menu-action', 'save-video');
                    }
                },
                { type: 'separator' },
                {
                    label: 'Exit',
                    accelerator: 'CmdOrCtrl+Q',
                    click: () => {
                        isQuitting = true;
                        app.quit();
                    }
                }
            ]
        },
        {
            label: 'Edit',
            submenu: [
                { role: 'undo' },
                { role: 'redo' },
                { type: 'separator' },
                { role: 'cut' },
                { role: 'copy' },
                { role: 'paste' }
            ]
        },
        {
            label: 'View',
            submenu: [
                { role: 'reload' },
                { role: 'forceReload' },
                { role: 'toggleDevTools' },
                { type: 'separator' },
                { role: 'resetZoom' },
                { role: 'zoomIn' },
                { role: 'zoomOut' },
                { type: 'separator' },
                { role: 'togglefullscreen' }
            ]
        },
        {
            label: 'Help',
            submenu: [
                {
                    label: 'Documentation',
                    click: () => {
                        shell.openExternal('https://github.com/Null8Void/Picture-Aliver');
                    }
                },
                {
                    label: 'About',
                    click: () => {
                        const { dialog } = require('electron');
                        dialog.showMessageBox(mainWindow, {
                            type: 'info',
                            title: 'About Picture-Aliver',
                            message: 'Picture-Aliver Desktop',
                            detail: 'Version 1.0.0\n\nAI Image-to-Video Generation Pipeline\n\nBuilt with Electron'
                        });
                    }
                }
            ]
        }
    ];
    
    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);
}

/**
 * Setup IPC handlers for renderer communication
 */
function setupIPC() {
    // Get backend URL
    ipcMain.handle('get-backend-url', () => {
        return `http://127.0.0.1:${BACKEND_PORT}`;
    });
    
    // Get app info
    ipcMain.handle('get-app-info', () => {
        return {
            version: app.getVersion(),
            platform: process.platform,
            arch: process.arch
        };
    });
    
    // Open folder dialog
    ipcMain.handle('open-folder-dialog', async (event, options) => {
        const { dialog } = require('electron');
        const result = await dialog.showOpenDialog(mainWindow, {
            properties: ['openDirectory'],
            ...options
        });
        return result;
    });
    
    // Open file dialog
    ipcMain.handle('open-file-dialog', async (event, options) => {
        const { dialog } = require('electron');
        const result = await dialog.showOpenDialog(mainWindow, {
            properties: ['openFile'],
            filters: [
                { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'webp'] }
            ],
            ...options
        });
        return result;
    });
    
    // Save file dialog
    ipcMain.handle('save-file-dialog', async (event, options) => {
        const { dialog } = require('electron');
        const result = await dialog.showSaveDialog(mainWindow, {
            filters: [
                { name: 'Video', extensions: ['mp4'] }
            ],
            ...options
        });
        return result;
    });
    
    // Show notification
    ipcMain.handle('show-notification', (event, { title, body }) => {
        const { Notification } = require('electron');
        new Notification({ title, body }).show();
    });
}

// =============================================================================
// APP LIFECYCLE
// =============================================================================

app.whenReady().then(() => {
    log.info('App ready, initializing...');
    
    // Start backend
    startBackend();
    
    // Setup IPC
    setupIPC();
    
    // Create window
    createWindow();
    
    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        } else if (mainWindow) {
            mainWindow.show();
        }
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('before-quit', () => {
    log.info('App quitting...');
    isQuitting = true;
    stopBackend();
});

app.on('quit', () => {
    log.info('App quit');
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
    log.error('Uncaught exception:', error);
});

process.on('unhandledRejection', (reason, promise) => {
    log.error('Unhandled rejection at:', promise, 'reason:', reason);
});