// Blog page functionality
let currentFilter = 'all';
let postsPerPage = 6;
let currentPage = 1;
let filteredPosts = [];

// Initialize blog page
document.addEventListener('DOMContentLoaded', function() {
  // Initialize theme
  initializeTheme();
  
  AOS.init({
    duration: 1000,
    easing: 'ease-out-cubic',
    once: true,
    offset: 50
  });

  // Initialize Lucide icons
  if (typeof lucide !== 'undefined') {
    lucide.createIcons();
  }

  // Load initial posts
  loadBlogPage();
  
  // Handle navbar scroll effect
  handleNavbarScroll();
});

// Load blog page content
function loadBlogPage() {
  filteredPosts = portfolioData.blogPosts;
  // Sort posts by date (most recent first)
  filteredPosts.sort((a, b) => new Date(b.date) - new Date(a.date));
  displayBlogPosts();
  updateLoadMoreButton();
}

// Filter posts by category
function filterPosts(category) {
  currentFilter = category;
  currentPage = 1;
  
  // Update active filter button
  document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
  event.target.closest('.filter-btn').classList.add('active');
  
  // Update filter description
  updateFilterDescription(category);
  
  // Filter posts
  if (category === 'all') {
    filteredPosts = portfolioData.blogPosts;
  } else if (category === 'research') {
    // Include both 'research' and 'project' types when filtering by research
    filteredPosts = portfolioData.blogPosts.filter(post => post.type === 'research' || post.type === 'project');
  } else {
    filteredPosts = portfolioData.blogPosts.filter(post => post.type === category);
  }
  
  // Sort posts by date (most recent first)
  filteredPosts.sort((a, b) => new Date(b.date) - new Date(a.date));
  
  displayBlogPosts();
  updateLoadMoreButton();
}

// Update filter description
function updateFilterDescription(category) {
  const descriptionContainer = document.getElementById('filter-description');
  const descriptions = {
    'all': '',
    'research': 'Here are experiments and research I\'ve conducted and wanted to share',
    'tutorial': 'Practical tutorials and guides',
    'literature': 'Scientific publications I\'ve read and studied, with my personal reflections'
  };
  
  const description = descriptions[category] || '';
  if (description) {
    descriptionContainer.innerHTML = `<p class="filter-description-text">${description}</p>`;
    descriptionContainer.style.display = 'block';
  } else {
    descriptionContainer.style.display = 'none';
  }
}

// Display blog posts
function displayBlogPosts() {
  const container = document.getElementById('all-blog-posts');
  const postsToShow = filteredPosts.slice(0, currentPage * postsPerPage);
  
  // Update container layout based on current filter
  container.classList.remove('literature-layout');
  if (currentFilter === 'literature') {
    container.classList.add('literature-layout');
  }
  
  container.innerHTML = '';
  
  postsToShow.forEach((post, index) => {
    const postCard = createBlogPostCardFull(post, index);
    container.appendChild(postCard);
  });
  
  // Re-initialize Lucide icons
  if (typeof lucide !== 'undefined') {
    lucide.createIcons();
  }
}

// Create blog post card for full blog page
function createBlogPostCardFull(post, index) {
  const card = document.createElement('div');
  card.className = 'blog-post-card-full';
  card.setAttribute('data-aos', 'fade-up');
  card.setAttribute('data-aos-delay', (index % postsPerPage * 100).toString());
  
  // Generate image HTML only if image exists, otherwise add spacing
  const imageHTML = post.image ? `
    <div class="blog-post-image">
      <img src="${post.image}" alt="${post.title}" loading="lazy">
    </div>
  ` : '<br>';
  
  card.innerHTML = `
    ${imageHTML}
    <div class="blog-post-content">
      <div class="blog-post-meta">
        <span class="blog-post-date">
          <i data-lucide="calendar"></i>
          ${formatDate(post.date)}
        </span>
        <span class="blog-post-read-time">
          <i data-lucide="clock"></i>
          ${post.readTime}
        </span>
        <span class="blog-post-type ${post.type}">${getTypeLabel(post.type)}</span>
      </div>
      
      <h3 class="blog-post-title">${post.title}</h3>
      <p class="blog-post-excerpt">${post.excerpt}</p>
      
      <div class="blog-post-tags">
        ${post.tags.map(tag => `<span class="blog-tag">${tag}</span>`).join('')}
      </div>
      
      <button class="blog-read-more" onclick="showBlogPostDetails(${post.id})">
        <span>Read More</span>
        <i data-lucide="arrow-right"></i>
      </button>
    </div>
  `;
  
  return card;
}

// Load more posts
function loadMorePosts() {
  currentPage++;
  displayBlogPosts();
  updateLoadMoreButton();
}

// Update load more button visibility
function updateLoadMoreButton() {
  const loadMoreBtn = document.getElementById('load-more-posts');
  const totalPages = Math.ceil(filteredPosts.length / postsPerPage);
  
  if (currentPage < totalPages) {
    loadMoreBtn.style.display = 'flex';
  } else {
    loadMoreBtn.style.display = 'none';
  }
}

// Get type label
function getTypeLabel(type) {
  const labels = {
    'research': 'Research',
    'project': 'Project',
    'tutorial': 'Tutorial',
    'literature': 'Literature',
    'news': 'News'
  };
  return labels[type] || type;
}

// Format date
function formatDate(dateString) {
  const options = { year: 'numeric', month: 'long', day: 'numeric' };
  return new Date(dateString).toLocaleDateString('en-US', options);
}

// Handle navbar scroll effect
function handleNavbarScroll() {
  const navbar = document.querySelector('.navbar');
  
  window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  });
}

// ============================================
// THEME MANAGEMENT (SHARED WITH MAIN SITE)
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

// Expose functions to global scope
window.toggleTheme = toggleTheme;
