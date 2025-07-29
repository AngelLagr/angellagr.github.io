// Global state
let currentTab = 'projects';
let showAllProjects = false;
let showAllCertificates = false;
const isMobile = window.innerWidth < 768;
const initialItems = isMobile ? 4 : 4; // Minimum 4 projects shown

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Initialize theme
    initializeTheme();
    
    // Initialize Lucide icons
    lucide.createIcons();
    
    // Initialize AOS animations
    AOS.init({
        duration: 1000,
        once: false,
        offset: 100
    });

    // Hide loading screen after a delay
    setTimeout(() => {
        const loadingScreen = document.getElementById('loading-screen');
        loadingScreen.classList.add('fade-out');
        setTimeout(() => {
            loadingScreen.style.display = 'none';
        }, 500);
    }, 500);

    // Initialize navbar
    initializeNavbar();
    
    // Initialize portfolio data
    initializePortfolio();
    
    // Initialize contact form
    initializeContactForm();
    
    // Initialize smooth scrolling
    initializeSmoothScroll();
    
    // Initialize typewriter effect (simplified)
    initializeTypewriter();
    
    // Load blog posts
    loadBlogPosts();
});

// Navbar functionality
function initializeNavbar() {
    const navToggle = document.getElementById('nav-toggle');
    const navCenter = document.getElementById('nav-center');
    const navLinks = document.querySelectorAll('.nav-link');

    // Mobile menu toggle
    navToggle.addEventListener('click', () => {
        navCenter.classList.toggle('active');
        navToggle.classList.toggle('active');
    });

    // Close mobile menu when clicking on a link
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            navCenter.classList.remove('active');
            navToggle.classList.remove('active');
        });
    });

    // Navbar scroll effect removed - keeping consistent background
}

// Portfolio functionality
function initializePortfolio() {
    loadProjects();
    loadTechStack();
    loadCertificates();
    updateShowMoreButtons();
    
    // Initialize Lucide icons
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
}

function loadProjects() {
    const container = document.getElementById('projects-container');
    const displayedProjects = showAllProjects ? 
        portfolioData.projects : 
        portfolioData.projects.slice(0, initialItems);

    container.innerHTML = displayedProjects.map(project => createProjectCard(project)).join('');
    
    // Reinitialize Lucide icons after content load
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
}

function loadCertificates() {
    const container = document.getElementById('certificates-container');
    if (!container) return;
    
    const itemsToShow = showAllCertificates ? portfolioData.certificates : portfolioData.certificates.slice(0, 3);
    container.innerHTML = itemsToShow.map(cert => createCertificateCard(cert)).join('');
    
    // Reinitialize Lucide icons after content load
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
}

function createCertificateCard(certificate) {
    return `
        <div class="certificate-card" data-aos="fade-up">
            <div class="certificate-content">
                <div class="certificate-status">${certificate.status}</div>
                <h3 class="certificate-title">${certificate.title}</h3>
                <p class="certificate-category">${certificate.category}</p>
                <p class="certificate-description">${certificate.description}</p>
                
                <div class="certificate-details">
                    <div class="detail-item">
                        <strong>Estimated Duration:</strong> ${certificate.estimatedDuration}
                    </div>
                </div>
            </div>
        </div>
    `;
}

function loadTechStack() {
    const container = document.getElementById('tech-stack-container');
    container.innerHTML = portfolioData.techStack.map(tech => createTechStackItem(tech)).join('');
    
    // Reinitialize Lucide icons after content load
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
}

function createProjectCard(project) {
    return `
        <div class="project-card" data-aos="fade-up">
            <img src="${project.image}" alt="${project.title}" class="project-image">
            <div class="project-content">
                <h3 class="project-title">${project.title}</h3>
                <p class="project-description">${project.description}</p>
                <div class="project-tech">
                    ${project.technologies.map(tech => `<span class="tech-tag">${tech}</span>`).join('')}
                </div>
                <div class="project-links">
                    ${project.github ? `
                        <a href="${project.github}" target="_blank" class="project-link">
                            <i data-lucide="github"></i>
                            Code
                        </a>
                    ` : ''}
                    ${project.demo ? `
                        <a href="${project.demo}" target="_blank" class="project-link">
                            <i data-lucide="external-link"></i>
                            Demo
                        </a>
                    ` : ''}
                    <button class="project-link details-btn" onclick="showProjectDetails(${project.id})">
                        <i data-lucide="info"></i>
                        Details
                    </button>
                </div>
            </div>
        </div>
    `;
}

function createCertificateCard(cert) {
    return `
        <div class="certificate-card" data-aos="fade-up">
            <div class="certificate-content">
                <h3 class="certificate-title">${cert.title}</h3>
                <p class="certificate-description">${cert.description}</p>
            </div>
        </div>
    `;
}

function createTechStackItem(tech) {
    return `
        <div class="tech-stack-item" data-aos="zoom-in">
            <img src="${tech.icon}" alt="${tech.name}" class="tech-icon">
            <span class="tech-name">${tech.name}</span>
        </div>
    `;
}

function switchTab(tabName) {
    // Update active tab button
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    event.target.closest('.tab-btn').classList.add('active');

    // Update active tab panel
    document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));
    document.getElementById(`${tabName}-tab`).classList.add('active');

    currentTab = tabName;

    // Reinitialize icons for new content
    setTimeout(() => {
        lucide.createIcons();
        AOS.refresh();
    }, 100);
}

function toggleShowMore(type) {
    if (type === 'projects') {
        showAllProjects = !showAllProjects;
        loadProjects();
        
        const btn = document.getElementById('show-more-projects');
        const span = btn.querySelector('span');
        const icon = btn.querySelector('i');
        
        span.textContent = showAllProjects ? 'See Less' : 'See More';
        icon.setAttribute('data-lucide', showAllProjects ? 'chevron-up' : 'chevron-down');
    } else {
        showAllCertificates = !showAllCertificates;
        loadCertificates();
        
        const btn = document.getElementById('show-more-certificates');
        const span = btn.querySelector('span');
        const icon = btn.querySelector('i');
        
        span.textContent = showAllCertificates ? 'See Less' : 'See More';
        icon.setAttribute('data-lucide', showAllCertificates ? 'chevron-up' : 'chevron-down');
    }

    // Update show more button visibility
    updateShowMoreButtons();

    // Reinitialize icons and animations
    setTimeout(() => {
        lucide.createIcons();
        AOS.refresh();
    }, 100);
}

function updateShowMoreButtons() {
    const projectsBtn = document.getElementById('show-more-projects');
    const certificatesBtn = document.getElementById('show-more-certificates');

    // Show/hide projects button
    if (portfolioData.projects.length <= initialItems) {
        projectsBtn.style.display = 'none';
    } else {
        projectsBtn.style.display = 'flex';
    }

    // Show/hide certificates button based on available research ideas
    if (portfolioData.certificates && portfolioData.certificates.length <= 3) {
        certificatesBtn.style.display = 'none';
    } else {
        certificatesBtn.style.display = 'flex';
    }
}

// TODO: Project Details Functionality
// Add this function to show detailed project information in a modal or expanded view
function showProjectDetails(projectId) {
    // Find the project data
    const project = portfolioData.projects.find(p => p.id === projectId);
    if (!project || !project.details) return;
    
    // TODO: Create modal or expanded section with:
    // 1. Project overview with rich text
    // 2. Mathematical formulations (consider MathJax for LaTeX)
    // 3. Architecture diagrams (SVG or images)
    // 4. Code snippets with syntax highlighting (Prism.js)
    // 5. Performance charts (Chart.js or D3.js)
    // 6. Research methodology section
    // 7. Results and analysis
    // 8. Image galleries for visualizations
    // 9. Interactive demos or simulations
    // 10. Related papers and references
    
    // Example modal structure:
    /*
    <div class="project-modal">
        <div class="modal-header">
            <h2>{project.title}</h2>
            <button class="close-btn">×</button>
        </div>
        <div class="modal-body">
            <div class="project-overview">
                <h3>Overview</h3>
                <p>{project.details.overview}</p>
            </div>
            <div class="methodology">
                <h3>Methodology</h3>
                <div class="math-formulas">
                    // LaTeX equations here
                </div>
                <div class="code-section">
                    // Syntax highlighted code
                </div>
            </div>
            <div class="results">
                <h3>Results</h3>
                <div class="charts">
                    // Performance charts
                </div>
                <div class="image-gallery">
                    // Result visualizations
                </div>
            </div>
        </div>
    </div>
    */
    
    console.log('Project details for:', project.title);
    console.log('Details:', project.details);
}

// Contact form functionality
function initializeContactForm() {
    const form = document.getElementById('contact-form');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(form);
        const data = {
            name: formData.get('name'),
            subject: formData.get('subject'),
            message: formData.get('message')
        };

        // Show loading state
        const submitBtn = form.querySelector('.submit-btn');
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<span>Sending...</span>';
        submitBtn.disabled = true;

        try {
            // Create mailto link
            const emailSubject = encodeURIComponent(`[Portfolio Contact] ${data.subject || 'New message'}`);
            const emailBody = encodeURIComponent(
                `Hi Angel,\n\n` +
                `You have received a new message from your portfolio contact form:\n\n` +
                `Name: ${data.name}\n` +
                `Subject: ${data.subject}\n\n` +
                `Message:\n${data.message}\n\n` +
                `---\n` +
                `This message was sent from your portfolio website.`
            );
            
            const mailtoLink = `mailto:angellagrange.contact@gmail.com?subject=${emailSubject}&body=${emailBody}`;
            
            // Open default email client
            window.location.href = mailtoLink;
            
            // Show success message after a short delay
            setTimeout(() => {
                Swal.fire({
                    title: 'Email Client Opened!',
                    text: 'Your default email client should open with the message pre-filled. Please send the email to complete your message.',
                    icon: 'success',
                    background: '#1a1a2e',
                    color: '#ffffff',
                    confirmButtonColor: '#6366f1'
                });
            }, 1000);

            form.reset();
        } catch (error) {
            // Show error message
            Swal.fire({
                title: 'Error!',
                text: 'There was an error opening your email client. Please send an email manually to angela.lagrasta@gmail.com',
                icon: 'error',
                background: '#1a1a2e',
                color: '#ffffff',
                confirmButtonColor: '#6366f1'
            });
        } finally {
            // Reset button
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
            lucide.createIcons();
        }
    });
}

// Smooth scrolling functionality
function initializeSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Utility function for smooth scrolling
function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

// Simple typewriter effect
function initializeTypewriter() {
    const elements = document.querySelectorAll('.typewriter');
    elements.forEach(element => {
        const text = element.textContent;
        element.textContent = '';
        
        let i = 0;
        const timer = setInterval(() => {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
            } else {
                clearInterval(timer);
            }
        }, 50);
    });
}

// Intersection Observer for animations
function initializeAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate');
            }
        });
    }, observerOptions);

    document.querySelectorAll('[data-animate]').forEach(el => {
        observer.observe(el);
    });
}

// Window resize handler
window.addEventListener('resize', () => {
    // Update mobile state
    const newIsMobile = window.innerWidth < 768;
    if (newIsMobile !== isMobile) {
        location.reload(); // Simple solution for responsive changes
    }
});

// Expose functions to global scope for HTML onclick handlers
window.switchTab = switchTab;
window.toggleShowMore = toggleShowMore;
window.scrollToSection = scrollToSection;
window.toggleTheme = toggleTheme;

// ============================================
// THEME MANAGEMENT
// ============================================

function initializeTheme() {
    // Check for saved theme or default to 'dark'
    const savedTheme = localStorage.getItem('theme') || 'dark';
    setTheme(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
}

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    
    // Update meta theme-color for mobile browsers
    const themeColorMeta = document.querySelector('meta[name="theme-color"]');
    if (themeColorMeta) {
        themeColorMeta.setAttribute('content', theme === 'light' ? '#ffffff' : '#0f0f0f');
    }
    
    // Reinitialize icons to ensure they display correctly
    if (typeof lucide !== 'undefined') {
        setTimeout(() => {
            lucide.createIcons();
        }, 100);
    }
}

// Project Details Modal Functions
function showProjectDetails(projectId) {
    const project = portfolioData.projects.find(p => p.id === projectId);
    if (!project || !project.details) return;

    const modal = createProjectDetailsModal(project);
    document.body.appendChild(modal);
    
    // Show modal with animation
    setTimeout(() => {
        modal.classList.add('show');
    }, 10);

    // Re-initialize Lucide icons for the modal
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
}

function createProjectDetailsModal(project) {
    const modal = document.createElement('div');
    modal.className = 'project-details-modal';
    modal.id = 'project-details-modal';
    
    modal.innerHTML = `
        <div class="project-details-content">
            <div class="project-details-header">
                <h2 class="project-details-title">${project.title}</h2>
                <button class="close-modal-btn" onclick="closeProjectDetails()">
                    <i data-lucide="x"></i>
                </button>
            </div>

            <div class="project-details-info">
                <div class="project-details-description">
                    ${project.description}
                </div>

                <div class="project-details-tech">
                    ${project.technologies.map(tech => `<span class="tech-tag">${tech}</span>`).join('')}
                </div>

                <div class="project-details-section">
                    <h3>Project Overview</h3>
                    <p>${project.details.overview}</p>
                </div>

                ${project.details.objectives && project.details.objectives.length > 0 ? `
                <div class="project-details-section">
                    <h3>Objectives</h3>
                    <ul>
                        ${project.details.objectives.map(obj => `<li>${obj}</li>`).join('')}
                    </ul>
                </div>
                ` : ''}

                ${project.details.methodology && project.details.methodology.trim() ? `
                <div class="project-details-section">
                    <h3>Methodology</h3>
                    <p>${project.details.methodology}</p>
                </div>
                ` : ''}

                ${project.details.results && project.details.results.trim() ? `
                <div class="project-details-section">
                    <h3>Results</h3>
                    <p>${project.details.results}</p>
                </div>
                ` : ''}

                ${project.details.challenges && project.details.challenges.trim() ? `
                <div class="project-details-section">
                    <h3>Technical Challenges</h3>
                    <p>${project.details.challenges}</p>
                </div>
                ` : ''}

                ${project.details.futureWork && project.details.futureWork.trim() ? `
                <div class="project-details-section">
                    <h3>Future Work</h3>
                    <p>${project.details.futureWork}</p>
                </div>
                ` : ''}
            </div>

            <div class="project-details-actions">
                ${project.github ? `
                    <a href="${project.github}" target="_blank" class="project-link">
                        <i data-lucide="github"></i>
                        View Code
                    </a>
                ` : ''}
                ${project.demo ? `
                    <a href="${project.demo}" target="_blank" class="project-link">
                        <i data-lucide="external-link"></i>
                        View Demo
                    </a>
                ` : ''}
                <button class="project-link" onclick="closeProjectDetails()">
                    <i data-lucide="arrow-left"></i>
                    Back
                </button>
            </div>
        </div>
    `;

    // Close modal when clicking outside
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeProjectDetails();
        }
    });

    // Close modal with Escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeProjectDetails();
        }
    });

    return modal;
}

function closeProjectDetails() {
    const modal = document.getElementById('project-details-modal');
    if (modal) {
        modal.classList.remove('show');
        setTimeout(() => {
            document.body.removeChild(modal);
        }, 300);
    }
}


// Blog Posts Functionality
function loadBlogPosts() {
    const container = document.getElementById('blog-posts');
    if (!container) return;
    
    // Sort posts by date (most recent first) and show only 2 most recent posts for home page grid
    const sortedPosts = [...portfolioData.blogPosts].sort((a, b) => new Date(b.date) - new Date(a.date));
    const recentPosts = sortedPosts.slice(0, 2);
    
    container.innerHTML = recentPosts.map(post => createBlogPostCardHome(post)).join('');
    
    // Reinitialize Lucide icons after content load
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
}

function createBlogPostCardHome(post) {
    const postDate = new Date(post.date).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
    
    return `
        <div class="blog-post-card-home" onclick="showBlogPostDetails(${post.id})">
            <div class="blog-post-meta-home">
                <span class="blog-post-date-home">
                    <i data-lucide="calendar"></i>
                    ${postDate}
                </span>
                <span class="blog-post-read-time-home">
                    <i data-lucide="clock"></i>
                    ${post.readTime}
                </span>
                <span class="blog-post-type-home ${post.type}">
                    ${post.type.charAt(0).toUpperCase() + post.type.slice(1)}
                </span>
            </div>
            <h3 class="blog-post-title-home">${post.title}</h3>
            <p class="blog-post-excerpt-home">${post.excerpt}</p>
            <div class="blog-post-tags-home">
                ${post.tags.slice(0, 3).map(tag => `<span class="blog-tag-home">${tag}</span>`).join('')}
            </div>
            <button class="blog-read-more-home" onclick="event.stopPropagation(); showBlogPostDetails(${post.id})">
                <span>Read More</span>
                <i data-lucide="arrow-right"></i>
            </button>
        </div>
    `;
}

function showBlogPostDetails(postId) {
    const post = portfolioData.blogPosts.find(p => p.id === postId);
    if (!post) return;
    
    const modal = document.createElement('div');
    modal.className = 'blog-post-modal';
    modal.innerHTML = createBlogPostModal(post);
    
    document.body.appendChild(modal);
    
    // Trigger animation
    setTimeout(() => {
        modal.classList.add('show');
    }, 10);
    
    // Add event listeners for closing
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeBlogPostModal();
        }
    });
    
    // Initialize Lucide icons in the modal
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
}

function createBlogPostModal(post) {
    const postDate = new Date(post.date).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
    
    return `
        <div class="blogmodal-content">
            <div class="blogmodal-header">
                <button class="close-blog-modal-btn" onclick="closeBlogPostModal()">
                    <i data-lucide="x"></i>
                </button>
            </div>
            
            <div class="blogmodal-body">
                ${post.image ? `
                <div class="blogmodal-hero">
                    <img src="${post.image}" alt="${post.title}" class="blogmodal-hero-image">
                </div>
                ` : '<br>'}
                
                <div class="blogmodal-type-badge ${post.type}">
                    ${post.type.charAt(0).toUpperCase() + post.type.slice(1)}
                </div>
                
                <div class="blogmodal-meta">
                    <span class="blogmodal-date">
                        <i data-lucide="calendar"></i>
                        ${postDate}
                    </span>
                    <span class="blogmodal-read-time">
                        <i data-lucide="clock"></i>
                        ${post.readTime}
                    </span>
                </div>
                
                <h1 class="blogmodal-title">${post.title}</h1>
                
                <div class="blogmodal-tags">
                    ${post.tags.map(tag => `<span class="blogmodal-tag">${tag}</span>`).join('')}
                </div>
                
                <div class="blogmodal-text">
                    ${post.content}
                </div>
            </div>
        </div>
    `;
}

function closeBlogPostModal() {
    const modal = document.querySelector('.blog-post-modal');
    if (modal) {
        modal.classList.remove('show');
        setTimeout(() => {
            document.body.removeChild(modal);
        }, 300);
    }
}
